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
    EXPECT_TRUE(record.leaves.empty());
    EXPECT_EQ(plan->getMemoryPlan().getCostComponents().retainedTapeBytes, 0u);
    EXPECT_TRUE(llvm::any_of(plan->getMemoryPlan().getSourceSelections(), [](const auto &selection) {
        return selection.candidates[selection.selectedCandidate].kind == AdResidualSourceKind::PrimalArgument;
    }));
}

TEST_F(VernonAutodiffTapePlanningTest, RuntimePolicyFallsBackToRematerializationUnderHardBudget) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @chain(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %square = arith.mulf %x, %x : f32
    %fourth = arith.mulf %square, %square : f32
    %result = arith.divf %fourth, %x : f32
    func.return %result : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("chain");
    function->setAttr("vernon.ad.planning_policy", StringAttr::get(&context, "min_runtime"));
    FailureOr<VernonAutodiffTapePlan> runtime = planFunction(*module, "chain", {"x"}, registry);
    ASSERT_TRUE(succeeded(runtime));
    EXPECT_TRUE(llvm::any_of(runtime->getMemoryPlan().getSourceSelections(), [](const auto &selection) {
        return selection.selectedCandidate < selection.candidates.size() &&
               selection.candidates[selection.selectedCandidate].kind == AdResidualSourceKind::StaticCapture;
    }));

    function->setAttr("vernon.ad.memory_budget_bytes", IntegerAttr::get(IntegerType::get(&context, 64), 8));
    FailureOr<VernonAutodiffTapePlan> budgeted = planFunction(*module, "chain", {"x"}, registry);
    ASSERT_TRUE(succeeded(budgeted));
    EXPECT_EQ(budgeted->getMemoryPlan().getCostComponents().retainedTapeBytes, 0u);
    EXPECT_TRUE(llvm::none_of(budgeted->getMemoryPlan().getSourceSelections(), [](const auto &selection) {
        return selection.selectedCandidate < selection.candidates.size() &&
               selection.candidates[selection.selectedCandidate].kind == AdResidualSourceKind::StaticCapture;
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
    EXPECT_TRUE(plan->getInvocationRecord().leaves.empty());
    auto recipeIt =
        llvm::find_if(plan->getMemoryPlan().getRematerializations(), [](const AdRematerializationRecipe &candidate) {
            return candidate.operations.size() == 1 && candidate.operations.front()->hasTrait<OpTrait::ConstantLike>();
        });
    ASSERT_NE(recipeIt, plan->getMemoryPlan().getRematerializations().end());
    const AdRematerializationRecipe &recipe = *recipeIt;
    ASSERT_EQ(recipe.operations.size(), 1u);
    EXPECT_TRUE(recipe.operations.front()->hasTrait<OpTrait::ConstantLike>());
    EXPECT_EQ(recipe.estimatedCost, 0u);
    EXPECT_TRUE(llvm::none_of(plan->getInvocationRecord().leaves,
                              [&](const AutodiffTapeLeaf &leaf) { return leaf.value == recipe.value; }));
    auto source =
        llvm::find_if(plan->getMemoryPlan().getSourceSelections(), [&](const AdResidualSourceSelection &selection) {
            return selection.key.value == recipe.value && selection.key.controlKind == AdControlSourceKind::None;
        });
    ASSERT_NE(source, plan->getMemoryPlan().getSourceSelections().end());
    ASSERT_LT(source->selectedCandidate, source->candidates.size());
    EXPECT_EQ(source->candidates[source->selectedCandidate].kind, AdResidualSourceKind::PureRematerialization);
    EXPECT_TRUE(plan->getRegions().empty());
    const AdMemoryPlan &memory = plan->getMemoryPlan();
    EXPECT_EQ(memory.getSelectedPolicy(), "min_memory");
    EXPECT_EQ(memory.getCostComponents().captureStoreBytes, memory.getCostComponents().backwardLoadBytes);
    EXPECT_EQ(memory.getCostComponents().retainedTapeBytes, 0u);
    EXPECT_GE(memory.getCostComponents().recomputationCost, recipe.estimatedCost);
    ASSERT_EQ(memory.getResiduals().size(), memory.getBufferAssignment().residualSlices.size());
    EXPECT_TRUE(llvm::any_of(plan->getMemoryPlan().getResiduals(), [&](const AdResidualInterval &residual) {
        return residual.value == recipe.value && residual.rematerialized && residual.byteSize == sizeof(float) &&
               residual.domain == AdMemoryDomain::PersistentResidual && residual.lifetimeEnd > residual.lifetimeBegin;
    }));
    for (auto [residual, slice] : llvm::zip_equal(memory.getResiduals(), memory.getBufferAssignment().residualSlices)) {
        EXPECT_GT(residual.lifetimeEnd, residual.lifetimeBegin);
        EXPECT_GT(residual.byteSize, 0u);
        if (residual.rematerialized) {
            EXPECT_EQ(slice.physicalBuffer, std::numeric_limits<unsigned>::max());
            EXPECT_EQ(slice.byteSize, 0u);
            continue;
        }
        EXPECT_EQ(residual.byteSize, slice.byteSize);
        ASSERT_LT(slice.physicalBuffer, memory.getBufferAssignment().physicalBuffers.size());
        const AdPhysicalBuffer &buffer = memory.getBufferAssignment().physicalBuffers[slice.physicalBuffer];
        EXPECT_EQ(slice.offset % residual.alignment, 0u);
        EXPECT_EQ(buffer.domain, residual.domain);
        EXPECT_GE(buffer.byteSize, slice.byteSize);
    }
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
    EXPECT_TRUE(record.leaves.empty());
    EXPECT_EQ(plan->getMemoryPlan().getCostComponents().retainedTapeBytes, 0u);
    EXPECT_EQ(llvm::count_if(plan->getMemoryPlan().getSourceSelections(),
                             [](const auto &selection) {
                                 return selection.candidates[selection.selectedCandidate].kind ==
                                        AdResidualSourceKind::PrimalArgument;
                             }),
              3u);
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

    EXPECT_TRUE(plan->getInvocationRecord().leaves.empty());
    EXPECT_TRUE(loop.record.leaves.empty());
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
    EXPECT_EQ(llvm::count_if(plan->getMemoryPlan().getSourceSelections(),
                             [&](const AdResidualSourceSelection &selection) {
                                 return selection.key.controlOperation == branch.operation &&
                                        selection.key.controlKind == AdControlSourceKind::Predicate &&
                                        selection.candidates[selection.selectedCandidate].kind ==
                                            AdResidualSourceKind::DynamicCapture;
                             }),
              1u);
}

TEST_F(VernonAutodiffTapePlanningTest, EnumeratesExactVersionReloadBeforeSelectingCurrentCapture) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @load_square(
      %input: !vernon.tensor_view<f32, [1], "read", "device"> {
        vernon.source_name = "input", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %zero = arith.constant 0 : index
    %loaded = "vernon.load"(%input, %zero)
        : (!vernon.tensor_view<f32, [1], "read", "device">, index) -> f32
    %result = arith.mulf %loaded, %loaded : f32
    func.return %result : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    FailureOr<VernonAutodiffTapePlan> plan = planFunction(*module, "load_square", {"input"}, registry);
    ASSERT_TRUE(succeeded(plan));
    LoadOp load;
    module->walk([&](LoadOp operation) { load = operation; });
    ASSERT_TRUE(load);
    auto source =
        llvm::find_if(plan->getMemoryPlan().getSourceSelections(), [&](const AdResidualSourceSelection &selection) {
            return selection.key.value == load.getResult();
        });
    ASSERT_NE(source, plan->getMemoryPlan().getSourceSelections().end());
    EXPECT_TRUE(llvm::any_of(source->candidates, [](const AdResidualSource &candidate) {
        return candidate.kind == AdResidualSourceKind::ExactVersionReload && candidate.legal &&
               !candidate.availableInCurrentContract && candidate.storageIdentity && candidate.versionBefore &&
               candidate.resourceReloadCost == 1 && candidate.deterministicReductionLegal;
    }));
    const AdResidualSource &selected = source->candidates[source->selectedCandidate];
    EXPECT_EQ(selected.kind, AdResidualSourceKind::StaticCapture);
    EXPECT_EQ(selected.captureStoreBytes, sizeof(float));
    EXPECT_EQ(selected.backwardLoadBytes, sizeof(float));
    EXPECT_EQ(plan->getMemoryPlan().getCostComponents().captureStoreBytes, sizeof(float));
    EXPECT_EQ(plan->getMemoryPlan().getCostComponents().backwardLoadBytes, sizeof(float));
    EXPECT_EQ(plan->getMemoryPlan().getMemoryBudgetBytes(), 64u * 1024u * 1024u);
    EXPECT_EQ(plan->getMemoryPlan().getSelectedPolicy(), "min_memory");

    func::FuncOp function = module->lookupSymbol<func::FuncOp>("load_square");
    function->setAttr("vernon.ad.planning_policy", StringAttr::get(&context, "balanced"));
    FailureOr<VernonAutodiffTapePlan> balanced = planFunction(*module, "load_square", {"input"}, registry);
    ASSERT_TRUE(succeeded(balanced));
    EXPECT_EQ(balanced->getMemoryPlan().getSelectedPolicy(), "balanced");
    function->setAttr("vernon.ad.planning_policy", StringAttr::get(&context, "min_runtime"));
    FailureOr<VernonAutodiffTapePlan> runtime = planFunction(*module, "load_square", {"input"}, registry);
    ASSERT_TRUE(succeeded(runtime));
    EXPECT_EQ(runtime->getMemoryPlan().getSelectedPolicy(), "min_runtime");
    function->setAttr("vernon.ad.memory_budget_bytes", IntegerAttr::get(IntegerType::get(&context, 64), 3));
    EXPECT_TRUE(failed(planFunction(*module, "load_square", {"input"}, registry)));
    function->removeAttr("vernon.ad.memory_budget_bytes");
    function->setAttr("vernon.ad.planning_policy", StringAttr::get(&context, "fastest"));
    EXPECT_TRUE(failed(planFunction(*module, "load_square", {"input"}, registry)));
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
    EXPECT_TRUE(record.leaves.empty());
    EXPECT_TRUE(llvm::any_of(plan->getMemoryPlan().getSourceSelections(), [](const auto &selection) {
        return selection.candidates[selection.selectedCandidate].kind == AdResidualSourceKind::PureRematerialization;
    }));
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

TEST(AutodiffBufferAssignmentTest, ReusesNonOverlappingIntervalsWithinMemoryDomains) {
    SmallVector<AdResidualInterval> residuals = {
        {Value{}, 0, AdMemoryDomain::PersistentResidual, 8, 8, 0, 5, false, 0},
        {Value{}, 1, AdMemoryDomain::PersistentResidual, 4, 4, 1, 4, false, 0},
        {Value{}, 2, AdMemoryDomain::PersistentResidual, 8, 8, 5, 9, false, 0},
        {Value{}, 3, AdMemoryDomain::PersistentResidual, 64, 8, 0, 9, true, 2},
        {Value{}, 4, AdMemoryDomain::TransientGradient, 8, 8, 5, 9, false, 0},
    };
    FailureOr<AdBufferAssignment> assignment = assignAdMemoryBuffers(residuals);
    ASSERT_TRUE(succeeded(assignment));
    ASSERT_EQ(assignment->physicalBuffers.size(), 3u);
    EXPECT_EQ(assignment->residualSlices[0].physicalBuffer, assignment->residualSlices[2].physicalBuffer);
    EXPECT_NE(assignment->residualSlices[0].physicalBuffer, assignment->residualSlices[1].physicalBuffer);
    EXPECT_EQ(assignment->residualSlices[3].physicalBuffer, std::numeric_limits<unsigned>::max());
    EXPECT_NE(assignment->residualSlices[2].physicalBuffer, assignment->residualSlices[4].physicalBuffer);
    EXPECT_EQ(assignment->physicalBuffers[0].offset, 0u);
    EXPECT_EQ(assignment->physicalBuffers[1].offset, 8u);
    EXPECT_EQ(assignment->physicalBuffers[2].offset, 0u);
    EXPECT_EQ(assignment->peakBytesByDomain[0], 12u);
    EXPECT_EQ(assignment->peakBytesByDomain[1], 8u);
    EXPECT_EQ(assignment->peakBytesByDomain[2], 0u);
    EXPECT_EQ(assignment->peakBytesByDomain[3], 0u);
    EXPECT_EQ(assignment->peakBytes, 20u);
}

TEST(AutodiffBufferAssignmentTest, RejectsInvalidIntervalsAndAlignment) {
    EXPECT_TRUE(failed(assignAdMemoryBuffers(
        {AdResidualInterval{Value{}, 0, AdMemoryDomain::PersistentResidual, 4, 3, 0, 1, false, 0}})));
    EXPECT_TRUE(failed(assignAdMemoryBuffers(
        {AdResidualInterval{Value{}, 0, AdMemoryDomain::PersistentResidual, 4, 4, 2, 1, false, 0}})));
}

TEST(AutodiffBufferAssignmentTest, SelectsRematerializationBySavedBytesPerCost) {
    SmallVector<AdResidualInterval> residuals = {
        {Value{}, 0, AdMemoryDomain::PersistentResidual, 8, 8, 0, 5, false, 0},
        {Value{}, 1, AdMemoryDomain::PersistentResidual, 4, 4, 1, 4, false, 0},
        {Value{}, 2, AdMemoryDomain::PersistentResidual, 8, 8, 5, 9, false, 0},
        {Value{}, 3, AdMemoryDomain::TransientGradient, 8, 8, 0, 9, false, 0},
    };
    SmallVector<AdRematerializationCandidate> candidates = {
        {{0, 2}, 8},
        {{1}, 1},
    };
    FailureOr<AdBudgetedBufferAssignment> assignment = assignAdMemoryBuffersWithinBudget(residuals, candidates, 16);
    ASSERT_TRUE(succeeded(assignment));
    EXPECT_FALSE(assignment->selectedCandidates[0]);
    EXPECT_TRUE(assignment->selectedCandidates[1]);
    EXPECT_EQ(assignment->recomputationCost, 1u);
    EXPECT_EQ(assignment->buffers.peakBytes, 16u);
}

TEST(AutodiffBufferAssignmentTest, RejectsUnreachableBudgetAndInvalidCandidates) {
    SmallVector<AdResidualInterval> residuals = {
        {Value{}, 0, AdMemoryDomain::PersistentResidual, 8, 8, 0, 5, false, 0},
        {Value{}, 1, AdMemoryDomain::PersistentResidual, 4, 4, 1, 4, false, 0},
    };
    EXPECT_TRUE(failed(assignAdMemoryBuffersWithinBudget(residuals, {AdRematerializationCandidate{{1}, 1}}, 4)));
    EXPECT_TRUE(failed(assignAdMemoryBuffersWithinBudget(residuals, {AdRematerializationCandidate{{0, 0}, 1}}, 8)));
    EXPECT_TRUE(failed(assignAdMemoryBuffersWithinBudget(
        residuals, {AdRematerializationCandidate{{0}, 1}, AdRematerializationCandidate{{0, 1}, 2}}, 8)));
    EXPECT_TRUE(failed(assignAdMemoryBuffersWithinBudget(residuals, {AdRematerializationCandidate{{2}, 1}}, 8)));
}

TEST(AutodiffBufferAssignmentTest, SelectsInteractingCandidatesThatOnlyReducePeakTogether) {
    SmallVector<AdResidualInterval> residuals = {
        {Value{}, 0, AdMemoryDomain::PersistentResidual, 8, 8, 0, 4, false, 0},
        {Value{}, 1, AdMemoryDomain::PersistentResidual, 8, 8, 4, 8, false, 0},
    };
    SmallVector<AdRematerializationCandidate> candidates = {
        {{0}, 1},
        {{1}, 1},
    };
    FailureOr<AdBudgetedBufferAssignment> assignment = assignAdMemoryBuffersWithinBudget(residuals, candidates, 0);
    ASSERT_TRUE(succeeded(assignment));
    EXPECT_EQ(assignment->selectedCandidates, SmallVector<bool>({true, true}));
    EXPECT_EQ(assignment->recomputationCost, 2u);
    EXPECT_EQ(assignment->buffers.peakBytes, 0u);
}

} // namespace
} // namespace mlir::vernon
