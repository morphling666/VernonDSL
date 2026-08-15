#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffAnalysis.h"
#include "mlir/Dialect/Vernon/Transforms/VernonAutodiffRules.h"
#include "mlir/Dialect/Vernon/Transforms/VernonGlobalIdIndexProof.h"
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

TEST_F(VernonAutodiffAnalysisTest, AppliesGlobalIdProofPolicies) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @proof_inputs(
      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"},
      %group: tensor<3xi32> {vernon.builtin = "workgroup_id"},
      %local: tensor<3xi32> {vernon.builtin = "local_invocation_id"},
      %unknown: index,
      %scalar_gid: index {vernon.builtin = "global_invocation_id"}) attributes {
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %four = arith.constant 4 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %group_x_i32 = tensor.extract %group[%zero] : tensor<3xi32>
    %local_x_i32 = tensor.extract %local[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_cast %gx_i32 : i32 to index
    %gx_bits = arith.bitcast %gx_i32 : i32 to i32
    %gx_bitcast = arith.index_cast %gx_bits : i32 to index
    %group_x = arith.index_castui %group_x_i32 : i32 to index
    %local_x = arith.index_castui %local_x_i32 : i32 to index
    %group_base = arith.muli %group_x, %four : index
    %reconstructed_x = arith.addi %group_base, %local_x : index
    %gy = arith.index_cast %gy_i32 : i32 to index
    %gz = arith.index_cast %gz_i32 : i32 to index
    %gx_unsigned = arith.index_castui %gx_i32 : i32 to index
    %remainder = arith.remui %gx, %four : index
    %quotient = arith.divui %gx, %four : index
    %short = arith.trunci %gx_i32 : i32 to i16
    %truncated = arith.index_cast %short : i16 to index
    func.return
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("proof_inputs");
    SmallVector<Value> casts;
    Value remainder;
    Value quotient;
    Value unsignedCast;
    function.walk([&](arith::IndexCastOp operation) { casts.push_back(operation.getResult()); });
    function.walk([&](arith::IndexCastUIOp operation) { unsignedCast = operation.getResult(); });
    function.walk([&](arith::RemUIOp operation) { remainder = operation.getResult(); });
    function.walk([&](arith::DivUIOp operation) { quotient = operation.getResult(); });
    ASSERT_EQ(casts.size(), 5u);
    ASSERT_TRUE(remainder);
    ASSERT_TRUE(quotient);
    ASSERT_TRUE(unsignedCast);

    SmallVector<Value> complete{casts[0], casts[2], casts[3]};
    EXPECT_TRUE(proveStrictInvocationOwnedIndex(complete));
    auto completeProof = proveInvocationOwnedIndex(complete);
    ASSERT_TRUE(completeProof);
    EXPECT_TRUE(completeProof->unitGridAxes.empty());
    EXPECT_FALSE(proveStrictInvocationOwnedIndex({unsignedCast}));
    EXPECT_FALSE(proveStrictInvocationOwnedIndex({casts[1]}));
    EXPECT_FALSE(proveStrictInvocationOwnedIndex({function.getArgument(4)}));
    EXPECT_FALSE(proveInvocationOwnedIndex({function.getArgument(4)}));

    arith::AddIOp reconstructed;
    function.walk([&](arith::AddIOp operation) { reconstructed = operation; });
    ASSERT_TRUE(reconstructed);
    EXPECT_FALSE(proveStrictInvocationOwnedIndex({reconstructed.getResult()}));
    EXPECT_FALSE(proveInvocationOwnedIndex({function.getArgument(2)}));
    EXPECT_FALSE(proveInvocationOwnedIndex({function.getArgument(1)}));

    SmallVector<Value> trailingUnknown{casts[0], casts[2], casts[3], function.getArgument(3)};
    EXPECT_FALSE(proveStrictInvocationOwnedIndex(trailingUnknown));
    EXPECT_TRUE(proveInvocationOwnedIndex(trailingUnknown));

    EXPECT_FALSE(proveInvocationOwnedIndex({function.getArgument(3), casts[0], casts[2], casts[3]}));
    auto partialProof = proveInvocationOwnedIndex({casts[0]});
    ASSERT_TRUE(partialProof);
    EXPECT_EQ(partialProof->unitGridAxes, (SmallVector<unsigned>{1, 2}));
    EXPECT_FALSE(proveInvocationOwnedIndex({function.getArgument(1)}));
    EXPECT_FALSE(proveInvocationOwnedIndex({casts[0], casts[2], casts[3], casts[0]}));
    EXPECT_FALSE(proveInvocationOwnedIndex({casts[0], casts[2], casts[3], remainder}));
    EXPECT_FALSE(proveInvocationOwnedIndex({casts[0], casts[2], casts[3], quotient}));
    EXPECT_FALSE(proveInvocationOwnedIndex({casts[0], casts[2], casts[3], casts[4]}));
}

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

TEST_F(VernonAutodiffAnalysisTest, DetectsActivityOutsideFirstAbiLeaf) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @aggregate(
      %input: tuple<f32, f32> {
        vernon.source_name = "input",
        vernon.abi_leaf_dtypes = ["f32", "f32"]
      }) -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %value = "vernon.tuple_get"(%input) {index = 1 : i64}
        : (tuple<f32, f32>) -> f32
    func.return %value : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("aggregate");
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(function, {"input.1"});
    ASSERT_TRUE(succeeded(analysis));

    EXPECT_TRUE(analysis->hasAnyActiveLeaf(function.getArgument(0)));
    EXPECT_FALSE(analysis->isActive(function.getArgument(0), 0));
    EXPECT_TRUE(analysis->isActive(function.getArgument(0), 1));
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
    EXPECT_TRUE(isa<scf::IfOp>(analysis->getRegions()[1].operation));
    ASSERT_TRUE(analysis->getRegions()[1].parentOrdinal);
    EXPECT_EQ(*analysis->getRegions()[1].parentOrdinal, 0u);
    BarrierOp barrier;
    function.walk([&](BarrierOp operation) { barrier = operation; });
    EXPECT_FALSE(analysis->isActive(barrier));
    EXPECT_TRUE(analysis->isActive(analysis->getRegions()[0].operation));
    EXPECT_TRUE(analysis->isActive(analysis->getRegions()[1].operation));
    auto barrierActivity = llvm::find_if(analysis->getOperations(), [&](const AutodiffOperationActivity &operation) {
        return operation.operation == barrier.getOperation();
    });
    ASSERT_NE(barrierActivity, analysis->getOperations().end());
    EXPECT_EQ(barrierActivity->effect, AutodiffEffectKind::Barrier);
    EXPECT_FALSE(barrierActivity->active);
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
    %device_old = "vernon.atomic"(%device, %index, %value) {
      atomic_kind = "add", ordering = "relaxed"
    } : (!vernon.tensor_view<f32, [1], "read_write", "device">, index, f32) -> f32
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
    EXPECT_EQ(llvm::count(effects, AutodiffEffectKind::Atomic), 2u);
    EXPECT_TRUE(llvm::is_contained(effects, AutodiffEffectKind::Barrier));
    EXPECT_FALSE(llvm::is_contained(effects, AutodiffEffectKind::ExternallyVisible));
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
    EXPECT_FALSE(analysis->isActive(store));
    auto storeActivity = llvm::find_if(analysis->getOperations(), [&](const AutodiffOperationActivity &operation) {
        return operation.operation == store.getOperation();
    });
    ASSERT_NE(storeActivity, analysis->getOperations().end());
    EXPECT_EQ(storeActivity->effect, AutodiffEffectKind::StorageWrite);
    EXPECT_FALSE(storeActivity->active);
    const AutodiffStorageIdentity *identity = analysis->getStorageIdentity(function.getArgument(1));
    ASSERT_NE(identity, nullptr);
    EXPECT_EQ(identity->internalAdjointOwnership, AutodiffInternalAdjointOwnership::None);
    EXPECT_FALSE(identity->externalGradientDestination);
}

TEST_F(VernonAutodiffAnalysisTest, ClassifiesActiveWorkgroupStorageAsCoupled) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @active_workgroup(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %workgroup: !vernon.tensor_view<f32, [1], "read_write", "workgroup">,
      %index: index) -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    "vernon.store"(%x, %workgroup, %index)
        : (f32, !vernon.tensor_view<f32, [1], "read_write", "workgroup">, index) -> ()
    %loaded = "vernon.load"(%workgroup, %index)
        : (!vernon.tensor_view<f32, [1], "read_write", "workgroup">, index) -> f32
    func.return %loaded : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("active_workgroup");
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(function, {"x"});
    ASSERT_TRUE(succeeded(analysis));
    const AutodiffStorageIdentity *identity = analysis->getStorageIdentity(function.getArgument(1));
    ASSERT_NE(identity, nullptr);
    EXPECT_EQ(identity->internalAdjointOwnership, AutodiffInternalAdjointOwnership::WorkgroupCoupled);
    EXPECT_FALSE(identity->externalGradientDestination);
}

TEST_F(VernonAutodiffAnalysisTest, RejectsUnprovenDeviceInternalAdjointOwnership) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @ambiguous_device_scratch(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %scratch: !vernon.tensor_view<f32, [1], "read_write", "device">)
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) attributes {
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %zero = arith.constant 0 : index
    "vernon.store"(%x, %scratch, %zero)
        : (f32, !vernon.tensor_view<f32, [1], "read_write", "device">, index) -> ()
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    %loaded = "vernon.load"(%scratch, %zero)
        : (!vernon.tensor_view<f32, [1], "read_write", "device">, index) -> f32
    func.return %loaded : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("ambiguous_device_scratch");
    EXPECT_NE(failureMessage(function, {"x"}).find("no proven lane-owned injective index mapping"), std::string::npos);
}

TEST_F(VernonAutodiffAnalysisTest, RejectsPartialGlobalIdTupleAsNonInjective) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @partial_global_id(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %scratch: !vernon.tensor_view<f32, [4], "read_write", "device">,
      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %zero = arith.constant 0 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gx = arith.index_cast %gx_i32 : i32 to index
    "vernon.store"(%x, %scratch, %gx)
        : (f32, !vernon.tensor_view<f32, [4], "read_write", "device">, index) -> ()
    %loaded = "vernon.load"(%scratch, %gx)
        : (!vernon.tensor_view<f32, [4], "read_write", "device">, index) -> f32
    func.return %loaded : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("partial_global_id");
    EXPECT_NE(failureMessage(function, {"x"}).find("no proven lane-owned injective index mapping"), std::string::npos);
}

TEST_F(VernonAutodiffAnalysisTest, RejectsScalarGlobalIdAsLanePrivate) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @scalar_global_id(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %scratch: !vernon.tensor_view<f32, [4], "read_write", "device">,
      %gid: index {vernon.builtin = "global_invocation_id"})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    "vernon.store"(%x, %scratch, %gid)
        : (f32, !vernon.tensor_view<f32, [4], "read_write", "device">, index) -> ()
    %loaded = "vernon.load"(%scratch, %gid)
        : (!vernon.tensor_view<f32, [4], "read_write", "device">, index) -> f32
    func.return %loaded : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("scalar_global_id");
    EXPECT_NE(failureMessage(function, {"x"}).find("no proven lane-owned injective index mapping"), std::string::npos);
}

TEST_F(VernonAutodiffAnalysisTest, AcceptsCompleteGlobalIdTupleAsLanePrivate) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @complete_global_id(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %scratch: !vernon.tensor_view<f32, [4, 4, 4], "read_write", "device">,
      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_cast %gx_i32 : i32 to index
    %gy = arith.index_cast %gy_i32 : i32 to index
    %gz = arith.index_cast %gz_i32 : i32 to index
    "vernon.store"(%x, %scratch, %gx, %gy, %gz)
        : (f32, !vernon.tensor_view<f32, [4, 4, 4], "read_write", "device">,
           index, index, index) -> ()
    %loaded = "vernon.load"(%scratch, %gx, %gy, %gz)
        : (!vernon.tensor_view<f32, [4, 4, 4], "read_write", "device">,
           index, index, index) -> f32
    func.return %loaded : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("complete_global_id");
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(function, {"x"});
    ASSERT_TRUE(succeeded(analysis));
    const AutodiffStorageIdentity *identity = analysis->getStorageIdentity(function.getArgument(1));
    ASSERT_NE(identity, nullptr);
    EXPECT_EQ(identity->internalAdjointOwnership, AutodiffInternalAdjointOwnership::LanePrivate);
}

TEST_F(VernonAutodiffAnalysisTest, AcceptsTrailingUnknownAfterCompleteGlobalIdTuple) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @complete_global_id_with_tail(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %scratch: !vernon.tensor_view<f32, [4, 4, 4, 4], "read_write", "device">,
      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"},
      %tail: index) -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_cast %gx_i32 : i32 to index
    %gy = arith.index_cast %gy_i32 : i32 to index
    %gz = arith.index_cast %gz_i32 : i32 to index
    "vernon.store"(%x, %scratch, %gx, %gy, %gz, %tail)
        : (f32, !vernon.tensor_view<f32, [4, 4, 4, 4], "read_write", "device">,
           index, index, index, index) -> ()
    %loaded = "vernon.load"(%scratch, %gx, %gy, %gz, %tail)
        : (!vernon.tensor_view<f32, [4, 4, 4, 4], "read_write", "device">,
           index, index, index, index) -> f32
    func.return %loaded : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("complete_global_id_with_tail");
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(function, {"x"});
    ASSERT_TRUE(succeeded(analysis));
    const AutodiffStorageIdentity *identity = analysis->getStorageIdentity(function.getArgument(1));
    ASSERT_NE(identity, nullptr);
    EXPECT_EQ(identity->internalAdjointOwnership, AutodiffInternalAdjointOwnership::LanePrivate);
}

TEST_F(VernonAutodiffAnalysisTest, RejectsRemainderAfterCompleteGlobalIdTuple) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @global_id_with_remainder(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %scratch: !vernon.tensor_view<f32, [4, 4, 4, 4], "read_write", "device">,
      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %four = arith.constant 4 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_cast %gx_i32 : i32 to index
    %gy = arith.index_cast %gy_i32 : i32 to index
    %gz = arith.index_cast %gz_i32 : i32 to index
    %wrapped = arith.remui %gx, %four : index
    "vernon.store"(%x, %scratch, %gx, %gy, %gz, %wrapped)
        : (f32, !vernon.tensor_view<f32, [4, 4, 4, 4], "read_write", "device">,
           index, index, index, index) -> ()
    %loaded = "vernon.load"(%scratch, %gx, %gy, %gz, %wrapped)
        : (!vernon.tensor_view<f32, [4, 4, 4, 4], "read_write", "device">,
           index, index, index, index) -> f32
    func.return %loaded : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("global_id_with_remainder");
    EXPECT_NE(failureMessage(function, {"x"}).find("no proven lane-owned injective index mapping"), std::string::npos);
}

TEST_F(VernonAutodiffAnalysisTest, RejectsCrossLaneDeviceInternalAdjointOwnership) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @proven_device_scratch(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %scratch: !vernon.tensor_view<f32, [4, 4, 4], "read_write", "device">,
      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"},
      %group: tensor<3xi32> {vernon.builtin = "workgroup_id"},
      %local: tensor<3xi32> {vernon.builtin = "local_invocation_id"})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) attributes {
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %group_x_i32 = tensor.extract %group[%zero] : tensor<3xi32>
    %local_x_i32 = tensor.extract %local[%zero] : tensor<3xi32>
    %gy = arith.index_cast %gy_i32 : i32 to index
    %gz = arith.index_cast %gz_i32 : i32 to index
    %group_x = arith.index_cast %group_x_i32 : i32 to index
    %local_x = arith.index_cast %local_x_i32 : i32 to index
    %four = arith.constant 4 : index
    %group_base = arith.muli %group_x, %four : index
    %store_x = arith.addi %group_base, %local_x : index
    "vernon.store"(%x, %scratch, %gz, %gy, %store_x)
        : (f32, !vernon.tensor_view<f32, [4, 4, 4], "read_write", "device">,
           index, index, index) -> ()
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    %next = arith.addi %local_x, %one : index
    %neighbor = arith.remui %next, %four : index
    %load_x = arith.addi %group_base, %neighbor : index
    %loaded = "vernon.load"(%scratch, %gz, %gy, %load_x)
        : (!vernon.tensor_view<f32, [4, 4, 4], "read_write", "device">,
           index, index, index) -> f32
    func.return %loaded : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("proven_device_scratch");
    EXPECT_NE(failureMessage(function, {"x"}).find("no proven lane-owned injective index mapping"), std::string::npos);
}

TEST_F(VernonAutodiffAnalysisTest, VersionsStorageEffectsAcrossBranches) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @storage_versions(
      %input: !vernon.tensor_view<f32, [1, 1, 1], "read", "device"> {
        vernon.source_name = "input", vernon.abi_leaf_dtypes = ["f32"]},
      %scratch: !vernon.tensor_view<f32, [1, 1, 1], "read_write", "device"> {
        vernon.source_name = "scratch", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1, 1, 1], "write", "device"> {
        vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %condition: i1,
      %gid: tensor<3xi32> {vernon.builtin = "global_invocation_id"}) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %gx_i32 = tensor.extract %gid[%zero] : tensor<3xi32>
    %gy_i32 = tensor.extract %gid[%one] : tensor<3xi32>
    %gz_i32 = tensor.extract %gid[%two] : tensor<3xi32>
    %gx = arith.index_cast %gx_i32 : i32 to index
    %gy = arith.index_cast %gy_i32 : i32 to index
    %gz = arith.index_cast %gz_i32 : i32 to index
    %value = "vernon.load"(%input, %gx, %gy, %gz)
        : (!vernon.tensor_view<f32, [1, 1, 1], "read", "device">,
           index, index, index) -> f32
    scf.if %condition {
      "vernon.store"(%value, %scratch, %gx, %gy, %gz)
          : (f32, !vernon.tensor_view<f32, [1, 1, 1], "read_write", "device">,
             index, index, index) -> ()
    } else {
      %twice = arith.addf %value, %value : f32
      "vernon.store"(%twice, %scratch, %gx, %gy, %gz)
          : (f32, !vernon.tensor_view<f32, [1, 1, 1], "read_write", "device">,
             index, index, index) -> ()
    }
    %stored = "vernon.load"(%scratch, %gx, %gy, %gz)
        : (!vernon.tensor_view<f32, [1, 1, 1], "read_write", "device">,
           index, index, index) -> f32
    "vernon.store"(%stored, %loss, %gx, %gy, %gz)
        : (f32, !vernon.tensor_view<f32, [1, 1, 1], "write", "device">,
           index, index, index) -> ()
    func.return
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("storage_versions");
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(function, {"input"}, {"loss"}, registry);
    ASSERT_TRUE(succeeded(analysis));

    EXPECT_EQ(analysis->getStorageIdentities().size(), 3u);
    EXPECT_EQ(analysis->getStorageEffects().size(), 5u);
    EXPECT_EQ(analysis->getStorageVersions().size(), 7u);
    const AutodiffStorageIdentity *inputIdentity = analysis->getStorageIdentity(function.getArgument(0));
    const AutodiffStorageIdentity *scratchIdentity = analysis->getStorageIdentity(function.getArgument(1));
    const AutodiffStorageIdentity *lossIdentity = analysis->getStorageIdentity(function.getArgument(2));
    ASSERT_NE(inputIdentity, nullptr);
    ASSERT_NE(scratchIdentity, nullptr);
    ASSERT_NE(lossIdentity, nullptr);
    EXPECT_EQ(inputIdentity->internalAdjointOwnership, AutodiffInternalAdjointOwnership::None);
    EXPECT_TRUE(inputIdentity->externalGradientDestination);
    EXPECT_EQ(scratchIdentity->internalAdjointOwnership, AutodiffInternalAdjointOwnership::LanePrivate);
    EXPECT_FALSE(scratchIdentity->externalGradientDestination);
    EXPECT_EQ(lossIdentity->internalAdjointOwnership, AutodiffInternalAdjointOwnership::None);
    EXPECT_FALSE(lossIdentity->externalGradientDestination);
    EXPECT_EQ(llvm::count_if(
                  analysis->getStorageVersions(),
                  [](const AutodiffStorageVersion &version) { return version.kind == StorageVersionKind::IfMerge; }),
              1u);
    for (const AutodiffStorageEffect &effect : analysis->getStorageEffects())
        EXPECT_TRUE(analysis->isActive(effect.operation));
    ASSERT_EQ(analysis->getLoads().size(), 2u);
    for (const AutodiffLoadInfo &load : analysis->getLoads()) {
        const AutodiffStorageEffect *effect = analysis->getStorageEffect(load.operation);
        ASSERT_NE(effect, nullptr);
        EXPECT_EQ(load.identity, effect->identity);
        EXPECT_EQ(load.versionBefore, effect->versionBefore);
        EXPECT_EQ(load.indices.size(), 3u);
        EXPECT_EQ(load.indexProvenance.size(), load.indices.size());
        EXPECT_TRUE(llvm::all_of(load.indexProvenance, [](AutodiffIndexProvenanceKind provenance) {
            return provenance == AutodiffIndexProvenanceKind::PureExpression;
        }));
        EXPECT_EQ(load.stability, load.identity == inputIdentity->id
                                      ? AutodiffStorageStabilityRequirement::RetainedExactVersion
                                      : AutodiffStorageStabilityRequirement::RestoreExactVersion);
        EXPECT_EQ(analysis->getLoadInfo(load.operation), &load);
    }
}

TEST_F(VernonAutodiffAnalysisTest, MergesZeroTripAndBackedgeStorageAtForExit) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @for_storage(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %scratch: !vernon.tensor_view<f32, [1], "read_write", "workgroup">
          {vernon.source_name = "scratch", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %zero = arith.constant 0 : index
    %four = arith.constant 4 : index
    %one = arith.constant 1 : index
    scf.for %index = %zero to %four step %one {
      "vernon.store"(%x, %scratch, %zero)
          : (f32, !vernon.tensor_view<f32, [1], "read_write", "workgroup">, index) -> ()
    }
    %result = "vernon.load"(%scratch, %zero)
        : (!vernon.tensor_view<f32, [1], "read_write", "workgroup">, index) -> f32
    func.return %result : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    FailureOr<VernonAutodiffAnalysisResult> analysis =
        analyzeAutodiffFunction(module->lookupSymbol<func::FuncOp>("for_storage"), {"x"});
    ASSERT_TRUE(succeeded(analysis));
    const AutodiffStorageVersion *phi = nullptr;
    const AutodiffStorageVersion *exit = nullptr;
    for (const AutodiffStorageVersion &version : analysis->getStorageVersions()) {
        if (version.kind == StorageVersionKind::ForPhi)
            phi = &version;
        if (version.kind == StorageVersionKind::ForExit)
            exit = &version;
    }
    ASSERT_NE(phi, nullptr);
    ASSERT_NE(exit, nullptr);
    ASSERT_EQ(phi->incomingVersions.size(), 2u);
    ASSERT_EQ(exit->incomingVersions.size(), 1u);
    EXPECT_EQ(exit->incomingVersions.front(), phi->id);
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

TEST_F(VernonAutodiffAnalysisTest, RejectsCollidingActiveAtomicOldValue) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @visible(
      %device: !vernon.tensor_view<f32, [1], "read_write", "device">
          {vernon.source_name = "device", vernon.abi_leaf_dtypes = ["f32"]},
      %x: f32,
      %index: index) -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) {
    %old = "vernon.atomic"(%device, %index, %x) {
      atomic_kind = "add", ordering = "relaxed"
    } : (!vernon.tensor_view<f32, [1], "read_write", "device">, index, f32) -> f32
    func.return %old : f32
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("visible");
    EXPECT_NE(failureMessage(function, {"device"}).find("requires a proven lane-exclusive device index mapping"),
              std::string::npos);
}

TEST_F(VernonAutodiffAnalysisTest, AnalyzesWorkgroupAtomicAddSideEffect) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @workgroup_atomic(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: index {vernon.builtin = "global_invocation_id"}) {
    %zero = arith.constant 0 : index
    %scratch = "vernon.workgroup_alloc"()
        : () -> !vernon.tensor_view<f32, [1], "read_write", "workgroup">
    %old = "vernon.atomic"(%scratch, %zero, %x) {
      atomic_kind = "add", ordering = "relaxed"
    } : (!vernon.tensor_view<f32, [1], "read_write", "workgroup">, index, f32) -> f32
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    %sum = "vernon.load"(%scratch, %zero)
        : (!vernon.tensor_view<f32, [1], "read_write", "workgroup">, index) -> f32
    "vernon.store"(%sum, %loss, %gid)
        : (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ()
    func.return
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("workgroup_atomic");
    VernonAutodiffRuleRegistry registry = createDefaultAutodiffRuleRegistry();
    FailureOr<VernonAutodiffAnalysisResult> analysis = analyzeAutodiffFunction(function, {"x"}, {"loss"}, registry);
    ASSERT_TRUE(succeeded(analysis));
    AtomicOp atomic;
    function.walk([&](AtomicOp operation) { atomic = operation; });
    ASSERT_TRUE(atomic);
    EXPECT_TRUE(analysis->isActive(atomic));
    const AutodiffStorageIdentity *identity = analysis->getStorageIdentity(atomic.getStorage());
    ASSERT_NE(identity, nullptr);
    EXPECT_EQ(identity->internalAdjointOwnership, AutodiffInternalAdjointOwnership::WorkgroupCoupled);
}

TEST_F(VernonAutodiffAnalysisTest, RejectsConflictingSsaAndResultDtypeMetadata) {
    OwningOpRef<ModuleOp> module = parse(R"mlir(
module {
  func.func @dtype_mismatch(
      %value: tuple<f32, i32> {
        vernon.source_name = "value",
        vernon.abi_leaf_dtypes = ["f32", "u32"]
      }) -> (tuple<f32, i32> {vernon.abi_leaf_dtypes = ["f32", "i32"]}) {
    func.return %value : tuple<f32, i32>
  }
}
)mlir");
    ASSERT_TRUE(module);
    func::FuncOp function = module->lookupSymbol<func::FuncOp>("dtype_mismatch");
    EXPECT_NE(failureMessage(function, {"value.0"}).find("ABI dtype metadata disagrees with its SSA value"),
              std::string::npos);
}

} // namespace
} // namespace mlir::vernon
