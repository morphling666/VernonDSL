#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffAnalysis.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffRules.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffTapePlanning.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

#include <limits>

namespace mlir::vernon {
namespace {

using Requirement = AutodiffPrimalRequirement;

class VernonAutodiffTapePlanningTest : public testing::Test {
protected:
    VernonAutodiffTapePlanningTest() {
        context.getOrLoadDialect<arith::ArithDialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<scf::SCFDialect>();
        context.getOrLoadDialect<VernonDialect>();
    }

    OwningOpRef<ModuleOp> parse(StringRef source) {
        return parseSourceString<ModuleOp>(source, ParserConfig(&context));
    }

    FailureOr<VernonAutodiffTapePlan> planFunction(ModuleOp module, StringRef name, ArrayRef<StringRef> wrt,
                                                   VernonAutodiffRuleRegistry &registry) {
        func::FuncOp function = module.lookupSymbol<func::FuncOp>(name);
        FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(function, wrt, registry);
        if (failed(analysis))
            return failure();
        return planAutodiffTape(function, *analysis, registry);
    }

    MLIRContext context;
};

TEST_F(VernonAutodiffTapePlanningTest, AddSavesNoPrimals) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @add(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %y: f32 {vernon.source_name = "y", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %result = arith.addf %x, %y : f32
    func.return %result : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    FailureOr<VernonAutodiffTapePlan> plan = planFunction(*module, "add", {"x"}, registry);
    ASSERT_TRUE(succeeded(plan));
    EXPECT_TRUE(plan->getInvocationRecord().leaves.empty());
    EXPECT_EQ(plan->getInvocationRecord().stride, 0u);
}

TEST_F(VernonAutodiffTapePlanningTest, MulAndDivSaveOnlyDeclaredValuesAndDeduplicate) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @arithmetic(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %y: f32 {vernon.source_name = "y", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %sum = arith.addf %x, %y : f32
    %product = arith.mulf %x, %x : f32
    %result = arith.divf %product, %y : f32
    func.return %result : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    FailureOr<VernonAutodiffTapePlan> plan = planFunction(*module, "arithmetic", {"x"}, registry);
    ASSERT_TRUE(succeeded(plan));

    const AutodiffTapeRecord &record = plan->getInvocationRecord();
    ASSERT_EQ(record.leaves.size(), 3u);
    EXPECT_EQ(record.leaves[0].value, module->lookupSymbol<func::FuncOp>("arithmetic").getArgument(0));
    EXPECT_EQ(record.leaves[1].value.getDefiningOp()->getName().getStringRef(), "arith.mulf");
    EXPECT_EQ(record.leaves[2].value, module->lookupSymbol<func::FuncOp>("arithmetic").getArgument(1));
    EXPECT_EQ(record.stride, 12u);
    EXPECT_TRUE(llvm::none_of(record.leaves, [](const AutodiffTapeLeaf &leaf) {
        return leaf.value.getDefiningOp() && leaf.value.getDefiningOp()->getName().getStringRef() == "arith.addf";
    }));
}

TEST_F(VernonAutodiffTapePlanningTest, NormalizesImplicitDtypesBeforeProvenanceComparison) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @implicit_dtype(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %y: f32 {vernon.source_name = "y", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %constant = arith.constant 2.0 : f32
    %scaled = arith.mulf %x, %constant : f32
    %result = arith.divf %scaled, %y : f32
    func.return %result : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    FailureOr<VernonAutodiffTapePlan> plan = planFunction(*module, "implicit_dtype", {"x"}, registry);
    ASSERT_TRUE(succeeded(plan));
    ASSERT_EQ(plan->getInvocationRecord().leaves.size(), 4u);
    EXPECT_TRUE(llvm::all_of(plan->getInvocationRecord().leaves,
                             [](const AutodiffTapeLeaf &leaf) { return leaf.dtype == "f32"; }));
}

TEST_F(VernonAutodiffTapePlanningTest, DecomposesAggregateThroughCanonicalAbiLeaves) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @aggregate(
      %value: tuple<f16, f64, f32> {
        vernon.source_name = "value",
        vernon.abi_leaf_dtypes = ["f16", "f64", "f32"]
      }) -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %result = "vernon.intrinsic"(%value) {name = "save_aggregate"}
        : (tuple<f16, f64, f32>) -> f32
    func.return %result : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    ASSERT_TRUE(succeeded(registry.registerRule(DifferentiationRule(
        "vernon.intrinsic.save_aggregate", 1, 1, {Requirement::operand(0)},
        [](Operation *, const AutodiffVjpBuildContext &, SmallVectorImpl<Value> &) { return success(); }, {},
        [](Operation *) { return success(); }))));

    FailureOr<VernonAutodiffTapePlan> plan = planFunction(*module, "aggregate", {"value"}, registry);
    ASSERT_TRUE(succeeded(plan));
    const AutodiffTapeRecord &record = plan->getInvocationRecord();
    ASSERT_EQ(record.leaves.size(), 3u);
    EXPECT_EQ(record.leaves[0].dtype, "f16");
    EXPECT_EQ(record.leaves[1].dtype, "f64");
    EXPECT_EQ(record.leaves[2].dtype, "f32");
    EXPECT_EQ(record.leaves[0].offset, 0u);
    EXPECT_EQ(record.leaves[1].offset, 8u);
    EXPECT_EQ(record.leaves[2].offset, 16u);
    EXPECT_EQ(record.stride, 24u);
    EXPECT_EQ(record.alignment, 8u);
}

TEST_F(VernonAutodiffTapePlanningTest, SeparatesInvocationAndNestedDynamicRecords) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @regions(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %factor: f32 {vernon.source_name = "factor", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %true = arith.constant true
    %result = scf.while (%before = %x) : (f32) -> (f32) {
      scf.condition(%true) %before : f32
    } do {
    ^bb0(%after: f32):
      %selected = scf.if %true -> (f32) {
        %product = arith.mulf %after, %factor : f32
        scf.yield %product : f32
      } else {
        scf.yield %after : f32
      }
      scf.yield %selected : f32
    }
    func.return %result : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    FailureOr<VernonAutodiffTapePlan> plan = planFunction(*module, "regions", {"x"}, registry);
    ASSERT_TRUE(succeeded(plan));
    ASSERT_EQ(plan->getRegions().size(), 2u);

    const AutodiffTapeRegion &loop = plan->getRegions()[0];
    const AutodiffTapeRegion &branch = plan->getRegions()[1];
    EXPECT_EQ(loop.parentRecord.kind, AutodiffParentRecordKind::Invocation);
    EXPECT_EQ(branch.parentRecord.kind, AutodiffParentRecordKind::DynamicRegion);
    ASSERT_TRUE(branch.parentRecord.regionOrdinal);
    EXPECT_EQ(*branch.parentRecord.regionOrdinal, loop.ordinal);
    EXPECT_EQ(branch.childOrdinal, 0u);
    ASSERT_EQ(loop.childRegionOrdinals, SmallVector<unsigned>({branch.ordinal}));
    EXPECT_TRUE(loop.control.executedCount);
    EXPECT_TRUE(loop.control.exitKind);
    EXPECT_TRUE(branch.control.predicate);

    ASSERT_EQ(plan->getInvocationRecord().leaves.size(), 1u);
    EXPECT_EQ(plan->getInvocationRecord().leaves.front().value,
              module->lookupSymbol<func::FuncOp>("regions").getArgument(1));
    ASSERT_EQ(loop.record.leaves.size(), 1u);
    EXPECT_TRUE(isa<BlockArgument>(loop.record.leaves.front().value));
    EXPECT_TRUE(branch.record.leaves.empty());

    EXPECT_TRUE(llvm::any_of(plan->getInvocationHeader().fields, [](const AutodiffTapeField &field) {
        return field.kind == AutodiffTapeFieldKind::RootRegionHandle;
    }));
    EXPECT_TRUE(llvm::any_of(loop.header.fields, [](const AutodiffTapeField &field) {
        return field.kind == AutodiffTapeFieldKind::ExecutedCount;
    }));
    EXPECT_EQ(loop.header.size, 40u);
    EXPECT_EQ(loop.header.size % loop.header.alignment, 0u);
    EXPECT_TRUE(llvm::any_of(branch.record.prefix.fields, [](const AutodiffTapeField &field) {
        return field.kind == AutodiffTapeFieldKind::Predicate;
    }));
    EXPECT_GT(plan->getStaticTapeBytesHint(), plan->getInvocationRecord().stride);
}

TEST_F(VernonAutodiffTapePlanningTest, RejectsRegistryDifferentFromAnalysis) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @multiply(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %result = arith.mulf %x, %x : f32
    func.return %result : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("multiply");
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(function, {"x"}, registry);
    ASSERT_TRUE(succeeded(analysis));
    VernonAutodiffRuleRegistry emptyRegistry;
    EXPECT_TRUE(failed(planAutodiffTape(function, *analysis, emptyRegistry)));
}

TEST_F(VernonAutodiffTapePlanningTest, PreservesProjectedLogicalDtypesFromSharedAnalysis) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @loop_dtype(
      %value: tuple<f32, i32> {
        vernon.source_name = "value",
        vernon.abi_leaf_dtypes = ["f32", "u32"]
      }) -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %true = arith.constant true
    %loop = scf.while (%before = %value) : (tuple<f32, i32>) -> tuple<f32, i32> {
      scf.condition(%true) %before : tuple<f32, i32>
    } do {
    ^bb0(%after: tuple<f32, i32>):
      %floating = "vernon.tuple_get"(%after) {index = 0 : i64}
          : (tuple<f32, i32>) -> f32
      %integer = "vernon.tuple_get"(%after) {index = 1 : i64}
          : (tuple<f32, i32>) -> i32
      %active = "vernon.intrinsic"(%floating, %integer) {name = "save_index"}
          : (f32, i32) -> f32
      %next = "vernon.tuple_create"(%active, %integer)
          : (f32, i32) -> tuple<f32, i32>
      scf.yield %next : tuple<f32, i32>
    }
    %result = "vernon.tuple_get"(%loop) {index = 0 : i64}
        : (tuple<f32, i32>) -> f32
    func.return %result : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    ASSERT_TRUE(succeeded(registry.registerRule(DifferentiationRule(
        "vernon.intrinsic.save_index", 2, 1, {Requirement::operand(1)},
        [](Operation *, const AutodiffVjpBuildContext &, SmallVectorImpl<Value> &) { return success(); }, {},
        [](Operation *) { return success(); }))));

    FailureOr<VernonAutodiffTapePlan> plan = planFunction(*module, "loop_dtype", {"value.0"}, registry);
    ASSERT_TRUE(succeeded(plan));
    ASSERT_EQ(plan->getRegions().size(), 1u);
    const AutodiffTapeRecord &record = plan->getRegions().front().record;
    ASSERT_EQ(record.leaves.size(), 1u);
    EXPECT_EQ(record.leaves.front().dtype, "u32");
}

TEST_F(VernonAutodiffTapePlanningTest, LaysOutNestedAggregateInsideDynamicRecord) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @nested_record(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %true = arith.constant true
    %result = scf.if %true -> (f32) {
      %wide = arith.constant 2.0 : f64
      %index = arith.constant 7 : i32
      %inner = "vernon.tuple_create"(%x, %wide) : (f32, f64) -> tuple<f32, f64>
      %nested = "vernon.tuple_create"(%inner, %index)
          : (tuple<f32, f64>, i32) -> tuple<tuple<f32, f64>, i32>
      %saved = "vernon.intrinsic"(%nested, %x) {name = "save_nested"}
          : (tuple<tuple<f32, f64>, i32>, f32) -> f32
      scf.yield %saved : f32
    } else {
      scf.yield %x : f32
    }
    func.return %result : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    ASSERT_TRUE(succeeded(registry.registerRule(DifferentiationRule(
        "vernon.intrinsic.save_nested", 2, 1, {Requirement::operand(0)},
        [](Operation *, const AutodiffVjpBuildContext &, SmallVectorImpl<Value> &contributions) {
            contributions.assign(2, Value{});
            return success();
        },
        {}, [](Operation *) { return success(); }))));
    FailureOr<VernonAutodiffTapePlan> plan = planFunction(*module, "nested_record", {"x"}, registry);
    ASSERT_TRUE(succeeded(plan));
    ASSERT_EQ(plan->getRegions().size(), 1u);
    const AutodiffTapeRecord &record = plan->getRegions().front().record;
    ASSERT_EQ(record.leaves.size(), 3u);
    EXPECT_EQ(record.leaves[0].path, "0.0");
    EXPECT_EQ(record.leaves[1].path, "0.1");
    EXPECT_EQ(record.leaves[2].path, "1");
    EXPECT_EQ(record.leaves[0].offset % record.leaves[0].alignment, 0u);
    EXPECT_EQ(record.leaves[1].offset % record.leaves[1].alignment, 0u);
    EXPECT_EQ(record.leaves[2].offset % record.leaves[2].alignment, 0u);
}

TEST(AutodiffTapeLayoutTest, UsesCheckedMixedAlignmentLayout) {
    FailureOr<AutodiffTapeLayout> layout =
        planAutodiffTapeLayout({AutodiffTapeSlot{2, 2}, AutodiffTapeSlot{8, 8}, AutodiffTapeSlot{4, 4}});
    ASSERT_TRUE(succeeded(layout));
    EXPECT_EQ(layout->offsets, SmallVector<uint64_t>({0, 8, 16}));
    EXPECT_EQ(layout->size, 20u);
    EXPECT_EQ(layout->stride, 24u);
    EXPECT_EQ(layout->alignment, 8u);
}

TEST(AutodiffTapeLayoutTest, RejectsOffsetSizeStrideAndAlignmentOverflow) {
    constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
    EXPECT_TRUE(failed(planAutodiffTapeLayout({AutodiffTapeSlot{8, 8}}, maximum, 1)));
    EXPECT_TRUE(failed(planAutodiffTapeLayout({AutodiffTapeSlot{maximum, 1}, AutodiffTapeSlot{1, 1}})));
    EXPECT_TRUE(failed(planAutodiffTapeLayout({AutodiffTapeSlot{maximum - 6, 8}})));
    EXPECT_TRUE(failed(planAutodiffTapeLayout({AutodiffTapeSlot{4, 3}})));
    EXPECT_TRUE(failed(planAutodiffTapeLayout({}, 0, 3)));
}

} // namespace
} // namespace mlir::vernon
