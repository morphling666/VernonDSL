#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffAnalysis.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

namespace mlir::vernon {
namespace {

class VernonAutodiffAnalysisTest : public testing::Test {
protected:
    VernonAutodiffAnalysisTest() {
        context.getOrLoadDialect<arith::ArithDialect>();
        context.getOrLoadDialect<cf::ControlFlowDialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<scf::SCFDialect>();
        context.getOrLoadDialect<VernonDialect>();
    }

    OwningOpRef<ModuleOp> parse(StringRef source) {
        return parseSourceString<ModuleOp>(source, ParserConfig(&context));
    }

    std::string failureMessage(func::FuncOp function, ArrayRef<StringRef> wrtPaths) {
        std::string message;
        ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
            llvm::raw_string_ostream stream(message);
            diagnostic.print(stream);
            return success();
        });
        EXPECT_TRUE(failed(analyzeAutodiffFunction(function, wrtPaths)));
        return message;
    }

    MLIRContext context;
};

TEST_F(VernonAutodiffAnalysisTest, PrunesValuesOutsideWrtToResultPath) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @activity(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %other: f32 {vernon.source_name = "other", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %constant = arith.constant 2.0 : f32
    %active = arith.mulf %x, %x : f32
    %inactive = arith.addf %other, %constant : f32
    func.return %active : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("activity");
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(function, {"x"});
    ASSERT_TRUE(succeeded(analysis));

    EXPECT_TRUE(analysis->isActive(function.getArgument(0), 0));
    EXPECT_FALSE(analysis->isActive(function.getArgument(1), 0));
    arith::AddFOp inactive;
    arith::MulFOp active;
    function.walk([&](arith::AddFOp operation) { inactive = operation; });
    function.walk([&](arith::MulFOp operation) { active = operation; });
    EXPECT_TRUE(analysis->isActive(active));
    EXPECT_FALSE(analysis->isActive(inactive));
    ASSERT_EQ(analysis->getWrtLeaves().size(), 1u);
    EXPECT_EQ(analysis->getWrtLeaves().front().path, "x");
    EXPECT_EQ(analysis->getActiveResultLeaves().front().path, "output");
}

TEST_F(VernonAutodiffAnalysisTest, ProjectsNestedAggregateLeavesCanonically) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @aggregate(
      %input: tuple<tuple<f32, i32>, f64> {
        vernon.source_name = "input",
        vernon.abi_leaf_dtypes = ["f32", "i32", "f64"]
      }) -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %inner = "vernon.tuple_get"(%input) {index = 0 : i64}
        : (tuple<tuple<f32, i32>, f64>) -> tuple<f32, i32>
    %value = "vernon.tuple_get"(%inner) {index = 0 : i64}
        : (tuple<f32, i32>) -> f32
    func.return %value : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("aggregate");
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(function, {"input.0"});
    ASSERT_TRUE(succeeded(analysis));

    ASSERT_EQ(analysis->getWrtLeaves().size(), 1u);
    EXPECT_EQ(analysis->getWrtLeaves().front().path, "input.0.0");
    EXPECT_TRUE(analysis->getWrtLeaves().front().derivativeType.isF32());
    EXPECT_TRUE(analysis->isActive(function.getArgument(0), 0));
    EXPECT_FALSE(analysis->isActive(function.getArgument(0), 2));
}

TEST_F(VernonAutodiffAnalysisTest, PropagatesStructLeavesThroughCreateAndGet) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  "vernon.struct"() {
    sym_name = "Pair",
    fields = ["left:f16", "count:i32"],
    abi_leaf_dtypes = ["f16", "i32"]
  } : () -> ()
  func.func @structure(
      %input: !vernon.struct<"Pair"> {
        vernon.source_name = "input",
        vernon.abi_leaf_dtypes = ["f16", "i32"]
      }) -> (f16 {vernon.abi_leaf_dtypes = ["f16"]}) {
    %left = "vernon.struct_get"(%input) {field = "left", index = 0 : i64}
        : (!vernon.struct<"Pair">) -> f16
    %count = "vernon.struct_get"(%input) {field = "count", index = 1 : i64}
        : (!vernon.struct<"Pair">) -> i32
    %copy = "vernon.struct_create"(%left, %count) {type_name = "Pair"}
        : (f16, i32) -> !vernon.struct<"Pair">
    %result = "vernon.struct_get"(%copy) {field = "left", index = 0 : i64}
        : (!vernon.struct<"Pair">) -> f16
    func.return %result : f16
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("structure");
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(function, {"input.left"});
    ASSERT_TRUE(succeeded(analysis));

    ASSERT_EQ(analysis->getWrtLeaves().size(), 1u);
    EXPECT_EQ(analysis->getWrtLeaves().front().path, "input.left");
    EXPECT_TRUE(analysis->getWrtLeaves().front().primalType.isF16());
    EXPECT_TRUE(analysis->getWrtLeaves().front().derivativeType.isF32());
    EXPECT_TRUE(analysis->isActive(function.getArgument(0), 0));
    EXPECT_FALSE(analysis->isActive(function.getArgument(0), 1));
}

TEST_F(VernonAutodiffAnalysisTest, ProjectsOnlyResultsActiveForSelectedWrt) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @result_projection(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %other: f64 {vernon.source_name = "other", vernon.abi_leaf_dtypes = ["f64"]})
      -> (tuple<f32, f64> {vernon.abi_leaf_dtypes = ["f32", "f64"]}) {
    %result = "vernon.tuple_create"(%x, %other)
        : (f32, f64) -> tuple<f32, f64>
    func.return %result : tuple<f32, f64>
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("result_projection");
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(function, {"x"});
    ASSERT_TRUE(succeeded(analysis));

    ASSERT_EQ(analysis->getActiveResultLeaves().size(), 1u);
    EXPECT_EQ(analysis->getActiveResultLeaves().front().path, "output.0");
    EXPECT_TRUE(analysis->getActiveResultLeaves().front().derivativeType.isF32());
}

TEST_F(VernonAutodiffAnalysisTest, DiscoversNestedStructuredRegions) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @regions(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %true = arith.constant true
    %result = scf.while (%before = %x) : (f32) -> (f32) {
      scf.condition(%true) %before : f32
    } do {
    ^bb0(%after: f32):
      %selected = scf.if %true -> (f32) {
        "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
        scf.yield %after : f32
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
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("regions");
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(function, {"x"});
    ASSERT_TRUE(succeeded(analysis));

    ASSERT_EQ(analysis->getRegions().size(), 2u);
    EXPECT_TRUE(isa<scf::WhileOp>(analysis->getRegions()[0].operation));
    EXPECT_FALSE(analysis->getRegions()[0].parentOrdinal);
    ASSERT_EQ(analysis->getRegions()[0].childOrdinals, SmallVector<unsigned>({1}));
    EXPECT_TRUE(isa<scf::IfOp>(analysis->getRegions()[1].operation));
    ASSERT_TRUE(analysis->getRegions()[1].parentOrdinal);
    EXPECT_EQ(*analysis->getRegions()[1].parentOrdinal, 0u);
    BarrierOp barrier;
    function.walk([&](BarrierOp operation) { barrier = operation; });
    EXPECT_TRUE(analysis->isActive(barrier));
    EXPECT_TRUE(analysis->isActive(analysis->getRegions()[0].operation));
    EXPECT_TRUE(analysis->isActive(analysis->getRegions()[1].operation));
    auto barrierActivity = llvm::find_if(analysis->getOperations(), [&](const AutodiffOperationActivity &operation) {
        return operation.operation == barrier.getOperation();
    });
    ASSERT_NE(barrierActivity, analysis->getOperations().end());
    EXPECT_EQ(barrierActivity->effect, AutodiffEffectKind::Barrier);
    EXPECT_TRUE(barrierActivity->active);
}

TEST_F(VernonAutodiffAnalysisTest, ClassifiesEffectsWithoutBackendPolicy) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @effects(
      %device: !vernon.tensor_view<f32, [1], "read_write", "device">,
      %workgroup: !vernon.tensor_view<f32, [1], "read_write", "workgroup">,
      %index: index, %value: f32) {
    %loaded = "vernon.load"(%device, %index)
        : (!vernon.tensor_view<f32, [1], "read_write", "device">, index) -> f32
    "vernon.store"(%value, %workgroup, %index)
        : (f32, !vernon.tensor_view<f32, [1], "read_write", "workgroup">, index) -> ()
    %old = "vernon.atomic"(%workgroup, %index, %value) {
      atomic_kind = "add", ordering = "relaxed"
    } : (!vernon.tensor_view<f32, [1], "read_write", "workgroup">, index, f32) -> f32
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    "vernon.store"(%loaded, %device, %index)
        : (f32, !vernon.tensor_view<f32, [1], "read_write", "device">, index) -> ()
    func.return
  }
}
)mlir");
    ASSERT_TRUE(module);
    SmallVector<AutodiffEffectKind> effects;
    module->walk([&](Operation *operation) {
        if (operation->getName().getDialectNamespace() == "vernon")
            effects.push_back(classifyAutodiffEffect(operation));
    });
    EXPECT_TRUE(llvm::is_contained(effects, AutodiffEffectKind::StorageRead));
    EXPECT_TRUE(llvm::is_contained(effects, AutodiffEffectKind::StorageWrite));
    EXPECT_TRUE(llvm::is_contained(effects, AutodiffEffectKind::Atomic));
    EXPECT_TRUE(llvm::is_contained(effects, AutodiffEffectKind::Barrier));
    EXPECT_TRUE(llvm::is_contained(effects, AutodiffEffectKind::ExternallyVisible));
}

TEST_F(VernonAutodiffAnalysisTest, KeepsEffectOnlyStructuredRegionActive) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @effect_region(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %workgroup: !vernon.tensor_view<f32, [1], "read_write", "workgroup">,
      %index: index) -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %true = arith.constant true
    scf.if %true {
      "vernon.store"(%x, %workgroup, %index)
          : (f32, !vernon.tensor_view<f32, [1], "read_write", "workgroup">, index) -> ()
    }
    func.return %x : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("effect_region");
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(function, {"x"});
    ASSERT_TRUE(succeeded(analysis));

    scf::IfOp region;
    StoreOp store;
    function.walk([&](scf::IfOp operation) { region = operation; });
    function.walk([&](StoreOp operation) { store = operation; });
    EXPECT_TRUE(analysis->isActive(region));
    EXPECT_TRUE(analysis->isActive(store));
    auto storeActivity = llvm::find_if(analysis->getOperations(), [&](const AutodiffOperationActivity &operation) {
        return operation.operation == store.getOperation();
    });
    ASSERT_NE(storeActivity, analysis->getOperations().end());
    EXPECT_EQ(storeActivity->effect, AutodiffEffectKind::StorageWrite);
    EXPECT_TRUE(storeActivity->active);
}

TEST_F(VernonAutodiffAnalysisTest, RejectsUnsupportedActiveOperation) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @unsupported(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %value = "vernon.intrinsic"(%x) {name = "unknown"}
        : (f32) -> f32
    func.return %value : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("unsupported");
    EXPECT_NE(failureMessage(function, {"x"}).find("active unsupported operation"), std::string::npos);
}

TEST_F(VernonAutodiffAnalysisTest, RejectsEmptyWrtAndMalformedDtypeMetadata) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @empty_wrt(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    func.return %x : f32
  }
  func.func @bad_metadata(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = [42 : i64]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    func.return %x : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    EXPECT_NE(
        failureMessage(module->lookupSymbol<func::FuncOp>("empty_wrt"), {}).find("requires at least one wrt path"),
        std::string::npos);
    EXPECT_NE(
        failureMessage(module->lookupSymbol<func::FuncOp>("bad_metadata"), {"x"}).find("malformed ABI dtype metadata"),
        std::string::npos);
}

TEST_F(VernonAutodiffAnalysisTest, RejectsMultipleNoncanonicalReturns) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @multiple_returns(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %condition: i1) -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    cf.cond_br %condition, ^left, ^right
  ^left:
    func.return %x : f32
  ^right:
    func.return %x : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    EXPECT_NE(failureMessage(module->lookupSymbol<func::FuncOp>("multiple_returns"), {"x"})
                  .find("requires one canonical return operation"),
              std::string::npos);
}

TEST_F(VernonAutodiffAnalysisTest, RejectsActiveExternallyVisibleEffect) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @visible(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %device: !vernon.tensor_view<f32, [1], "write", "device">,
      %index: index) -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %constant = arith.constant 1.0 : f32
    "vernon.store"(%constant, %device, %index)
        : (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ()
    func.return %constant : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("visible");
    EXPECT_NE(failureMessage(function, {"x"}).find("active externally visible operation"), std::string::npos);
}

} // namespace
} // namespace mlir::vernon
