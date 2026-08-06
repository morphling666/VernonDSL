#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerCPUAutodiff.h"
#include "mlir/Dialect/Vernon/Transforms/VernonStructuredVjp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include <gtest/gtest.h>

namespace mlir::vernon {
namespace {

class VernonStructuredVjpTest : public testing::Test {
protected:
    VernonStructuredVjpTest() {
        context.getOrLoadDialect<arith::ArithDialect>();
        context.getOrLoadDialect<func::FuncDialect>();
        context.getOrLoadDialect<LLVM::LLVMDialect>();
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
            buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                               StructuredVjpOptions{{"x", "y"}, "forward" + suffix, "backward" + suffix});
        ASSERT_TRUE(succeeded(result)) << testCase.operation;
        EXPECT_TRUE(succeeded(verify(*module))) << testCase.operation;
        EXPECT_EQ(result->forward.getNumArguments(), 2u);
        EXPECT_EQ(result->forward.getNumResults(), 1u);
        EXPECT_TRUE(isa<TupleType>(result->forward.getResultTypes().front()));
        EXPECT_EQ(result->backward.getNumArguments(), 3u);
        EXPECT_EQ(result->backward.getNumResults(), 1u);
        EXPECT_TRUE(isa<TupleType>(result->backward.getResultTypes().front()));
        ASSERT_EQ(result->derivativeRules.size(), 1u);
        EXPECT_EQ(result->derivativeRules.front(), testCase.operation);
        auto forwardResult = cast<TupleType>(result->forward.getResultTypes().front());
        EXPECT_TRUE(isa<AdTapeType>(forwardResult.getType(1)));
        EXPECT_TRUE(isa<AdRegionHeaderType>(forwardResult.getType(2)));
        unsigned writes = 0;
        result->forward.walk([&](AdWriteLeafOp) { ++writes; });
        EXPECT_EQ(writes, testCase.tapeLeaves + 1);
        for (func::FuncOp profile : {result->forward, result->backward}) {
            for (unsigned argument = 0; argument < profile.getNumArguments(); ++argument)
                EXPECT_FALSE(profile.getArgAttr(argument, "vernon.builtin"));
            profile.walk([&](Operation *operation) {
                EXPECT_NE(operation->getName().getDialectNamespace(), LLVM::LLVMDialect::getDialectNamespace());
            });
        }
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
        buildStructuredVjp(powerModule->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"x", "y"}, "power_forward", "power_backward"});
    ASSERT_TRUE(succeeded(power));
    EXPECT_EQ(power->derivativeRules, SmallVector<std::string>({"vernon.intrinsic.pow"}));
    unsigned powerWrites = 0;
    power->forward.walk([&](AdWriteLeafOp) { ++powerWrites; });
    EXPECT_EQ(powerWrites, 4u);
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
    FailureOr<StructuredVjpResult> result = buildStructuredVjp(
        module->lookupSymbol<func::FuncOp>("primal"), StructuredVjpOptions{{"x"}, "shared_forward", "shared_backward"});
    ASSERT_TRUE(succeeded(result));
    unsigned writes = 0;
    result->forward.walk([&](AdWriteLeafOp) { ++writes; });
    EXPECT_EQ(writes, 2u);
    EXPECT_EQ(result->derivativeRules, SmallVector<std::string>({"arith.addf", "arith.mulf"}));
    unsigned additions = 0;
    result->backward.walk([&](arith::AddFOp) { ++additions; });
    EXPECT_GE(additions, 3u);
}

TEST_F(VernonStructuredVjpTest, CpuPreparationRemovesAllLogicalTapeHandles) {
    OwningOpRef<ModuleOp> module = parse("arith.mulf");
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"x", "y"}, "prepared_forward", "prepared_backward"});
    ASSERT_TRUE(succeeded(result));

    PassManager manager(&context);
    manager.addPass(createVernonPrepareCPUAutodiffSignaturesPass());
    manager.addPass(createVernonLowerCPUAutodiffPass());
    ASSERT_TRUE(succeeded(manager.run(*module)));
    EXPECT_TRUE(succeeded(verify(*module)));
    unsigned typedCallbacks = 0;
    unsigned requiredByteReads = 0;
    unsigned prematureLlvmCalls = 0;
    module->walk([&](Operation *operation) {
        typedCallbacks += isa<CpuAdCallbackOp>(operation);
        requiredByteReads += isa<CpuAdRequiredBytesOp>(operation);
        prematureLlvmCalls += isa<LLVM::CallOp>(operation);
    });
    EXPECT_GT(typedCallbacks, 0u);
    EXPECT_GT(requiredByteReads, 0u);
    EXPECT_EQ(prematureLlvmCalls, 0u);

    PassManager conversion(&context);
    conversion.addPass(createVernonCPUAutodiffToLLVMPass());
    ASSERT_TRUE(succeeded(conversion.run(*module)));
    EXPECT_TRUE(succeeded(verify(*module)));

    bool hasLogicalOperation = false;
    unsigned indirectCallbacks = 0;
    module->walk([&](Operation *operation) {
        hasLogicalOperation = hasLogicalOperation ||
                              isa<AdCaptureOp, AdBeginInvocationOp, AdBeginRegionOp, AdReserveRecordOp, AdWriteLeafOp,
                                  AdEndRegionOp, AdCommitOp, AdReadRecordOffsetOp, AdReadExecutedCountOp,
                                  AdReadExitKindOp, AdReadNestedRegionOp, AdReadLeafOp>(operation);
        EXPECT_FALSE(isa<CpuAdCallbackOp>(operation));
        EXPECT_FALSE(isa<CpuAdRequiredBytesOp>(operation));
        for (Type type : operation->getOperandTypes())
            EXPECT_FALSE(containsLogicalAutodiffHandle(type));
        for (Type type : operation->getResultTypes())
            EXPECT_FALSE(containsLogicalAutodiffHandle(type));
        if (auto call = dyn_cast<LLVM::CallOp>(operation)) {
            ++indirectCallbacks;
            EXPECT_FALSE(call.getCallee().has_value());
        }
        for (NamedAttribute attribute : operation->getAttrs())
            if (auto symbol = dyn_cast<FlatSymbolRefAttr>(attribute.getValue()))
                EXPECT_FALSE(symbol.getValue().starts_with("__vernon_cpu_ad_"));
    });
    EXPECT_FALSE(hasLogicalOperation);
    EXPECT_GT(indirectCallbacks, 0u);
}

TEST_F(VernonStructuredVjpTest, RejectsMalformedTypedCpuCallbackSignature) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @malformed(%descriptor: index, %payload: f32) {
    "vernon.cpu_ad.callback"(%descriptor, %payload) {callback = 1 : i32} : (index, f32) -> ()
    func.return
  }
}
)mlir",
                                                               &context);
    ASSERT_TRUE(module);
    PassManager conversion(&context);
    conversion.addPass(createVernonCPUAutodiffToLLVMPass());
    EXPECT_TRUE(failed(conversion.run(*module)));
}

TEST_F(VernonStructuredVjpTest, RejectsUnsupportedTypedCpuCallbackPayload) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @unsupported_payload(%descriptor: index, %payload: f128) {
    %zero = arith.constant 0 : i64
    "vernon.cpu_ad.callback"(%descriptor, %zero, %zero, %payload, %zero)
        {callback = 3 : i32} : (index, i64, i64, f128, i64) -> ()
    func.return
  }
}
)mlir",
                                                               &context);
    ASSERT_TRUE(module);
    PassManager conversion(&context);
    conversion.addPass(createVernonCPUAutodiffToLLVMPass());
    EXPECT_TRUE(failed(conversion.run(*module)));
}

TEST_F(VernonStructuredVjpTest, RejectsAmbiguousOptionsWithoutMutatingPrimal) {
    for (const StructuredVjpOptions &options :
         {StructuredVjpOptions{{"x"}, "derivative", "derivative"},
          StructuredVjpOptions{{"x", "x"}, "forward", "backward"}, StructuredVjpOptions{{""}, "forward", "backward"}}) {
        OwningOpRef<ModuleOp> module = parse("arith.addf");
        ASSERT_TRUE(module);
        func::FuncOp primal = module->lookupSymbol<func::FuncOp>("primal");
        EXPECT_TRUE(failed(buildStructuredVjp(primal, options)));
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
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"x"}, "structured_forward", "structured_backward"});
    ASSERT_TRUE(succeeded(result));
    unsigned writes = 0;
    result->forward.walk([&](AdWriteLeafOp) { ++writes; });
    EXPECT_EQ(writes, 3u);
    EXPECT_GT(result->tapeBytes, 0u);
    EXPECT_TRUE(result->backward.getResultTypes().front().isF32());
    auto cotangentDtypes = cast<ArrayAttr>(result->backward.getArgAttr(2, "vernon.abi_leaf_dtypes"));
    ASSERT_EQ(cotangentDtypes.size(), 1u);
    EXPECT_EQ(cast<StringAttr>(cotangentDtypes[0]).getValue(), "f32");
    auto gradientDtypes = cast<ArrayAttr>(result->backward.getResultAttr(0, "vernon.abi_leaf_dtypes"));
    ASSERT_EQ(gradientDtypes.size(), 1u);
    EXPECT_EQ(cast<StringAttr>(gradientDtypes[0]).getValue(), "f32");
}

TEST_F(VernonStructuredVjpTest, ProjectsTensorAndStructLeavesThroughCanonicalAbi) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  "vernon.struct"() {sym_name = "Parameters", fields = ["factor:f16", "bias:f64"],
      abi_leaf_dtypes = ["f16", "f64"]} : () -> ()
  func.func @primal(
      %value: tensor<2xf16> {vernon.source_name = "value", vernon.abi_leaf_dtypes = ["f16"]},
      %parameters: !vernon.struct<"Parameters"> {
          vernon.source_name = "parameters", vernon.abi_leaf_dtypes = ["f16", "f64"]})
      -> (tuple<f16, f64> {vernon.abi_leaf_dtypes = ["f16", "f64"]}) attributes {vernon.entry} {
    %factor = "vernon.struct_get"(%parameters) {field = "factor", index = 0 : i64}
        : (!vernon.struct<"Parameters">) -> f16
    %bias = "vernon.struct_get"(%parameters) {field = "bias", index = 1 : i64}
        : (!vernon.struct<"Parameters">) -> f64
    %scale = tensor.splat %factor : tensor<2xf16>
    %scaled = arith.mulf %value, %scale : tensor<2xf16>
    %zero = arith.constant 0 : index
    %first = tensor.extract %scaled[%zero] : tensor<2xf16>
    %squared_bias = arith.mulf %bias, %bias : f64
    %result = "vernon.tuple_create"(%first, %squared_bias) : (f16, f64) -> tuple<f16, f64>
    func.return %result : tuple<f16, f64>
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result = buildStructuredVjp(
        module->lookupSymbol<func::FuncOp>("primal"),
        StructuredVjpOptions{
            {"value", "parameters.factor", "parameters.bias"}, "aggregate_forward", "aggregate_backward"});
    ASSERT_TRUE(succeeded(result));
    ASSERT_TRUE(succeeded(verify(*module)));
    EXPECT_TRUE(result->backward.getArgument(2).getType().isF32());
    EXPECT_TRUE(result->backward.getArgument(3).getType().isF64());
    auto gradients = dyn_cast<TupleType>(result->backward.getResultTypes().front());
    ASSERT_TRUE(gradients);
    ASSERT_EQ(gradients.size(), 3u);
    EXPECT_TRUE(gradients.getType(0).isF64());
    EXPECT_TRUE(gradients.getType(1).isF32());
    EXPECT_EQ(gradients.getType(2), RankedTensorType::get({2}, Float32Type::get(&context)));
    auto gradientDtypes = cast<ArrayAttr>(result->backward.getResultAttr(0, "vernon.abi_leaf_dtypes"));
    ASSERT_EQ(gradientDtypes.size(), 3u);
    EXPECT_EQ(cast<StringAttr>(gradientDtypes[0]).getValue(), "f64");
    EXPECT_EQ(cast<StringAttr>(gradientDtypes[1]).getValue(), "f32");
    EXPECT_EQ(cast<StringAttr>(gradientDtypes[2]).getValue(), "f32");
}

TEST_F(VernonStructuredVjpTest, ProjectsNestedAggregateOutputCotangentsAndDynamicTensorIndices) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  "vernon.struct"() {sym_name = "Pair", fields = ["selected:f32", "values:tensor<2xf32>"],
      abi_leaf_dtypes = ["f32", "f32"]} : () -> ()
  func.func @primal(
      %values: tensor<2xf32> {vernon.source_name = "values", vernon.abi_leaf_dtypes = ["f32"]},
      %index: i32 {vernon.source_name = "index", vernon.abi_leaf_dtypes = ["i32"]})
      -> (!vernon.struct<"Pair"> {vernon.abi_leaf_dtypes = ["f32", "f32"]}) attributes {vernon.entry} {
    %dynamic_index = arith.index_cast %index : i32 to index
    %selected = tensor.extract %values[%dynamic_index] : tensor<2xf32>
    %squared = arith.mulf %values, %values : tensor<2xf32>
    %result = "vernon.struct_create"(%selected, %squared) {type_name = "Pair"}
        : (f32, tensor<2xf32>) -> !vernon.struct<"Pair">
    func.return %result : !vernon.struct<"Pair">
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"values"}, "aggregate_output_forward", "aggregate_output_backward"});
    ASSERT_TRUE(succeeded(result));
    ASSERT_TRUE(succeeded(verify(*module)));
    ASSERT_EQ(result->backward.getNumArguments(), 4u);
    EXPECT_TRUE(result->backward.getArgument(2).getType().isF32());
    EXPECT_EQ(result->backward.getArgument(3).getType(), RankedTensorType::get({2}, Float32Type::get(&context)));
    EXPECT_EQ(result->backward.getResultTypes().front(), RankedTensorType::get({2}, Float32Type::get(&context)));
}

TEST_F(VernonStructuredVjpTest, FunctionalizesTensorViewLoadAndStoreVjp) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %values: !vernon.tensor_view<f32, [3], "read_write", "device"> {
          vernon.source_name = "values", vernon.element_abi_leaf_dtypes = ["f32"]},
      %scale: f32 {vernon.source_name = "scale", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) attributes {vernon.entry} {
    %one_i32 = arith.constant 1 : i32
    %one = arith.index_cast %one_i32 : i32 to index
    %old = "vernon.load"(%values, %one)
        : (!vernon.tensor_view<f32, [3], "read_write", "device">, index) -> f32
    %updated = arith.mulf %old, %scale : f32
    "vernon.store"(%updated, %values, %one)
        : (f32, !vernon.tensor_view<f32, [3], "read_write", "device">, index) -> ()
    %zero_i32 = arith.constant 0 : i32
    %zero = arith.index_cast %zero_i32 : i32 to index
    %first = "vernon.load"(%values, %zero)
        : (!vernon.tensor_view<f32, [3], "read_write", "device">, index) -> f32
    %current = "vernon.load"(%values, %one)
        : (!vernon.tensor_view<f32, [3], "read_write", "device">, index) -> f32
    %result = arith.addf %first, %current : f32
    func.return %result : f32
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"scale", "values"}, "storage_forward", "storage_backward"});
    ASSERT_TRUE(succeeded(result));
    ASSERT_TRUE(succeeded(verify(*module)));
    auto gradients = dyn_cast<TupleType>(result->backward.getResultTypes().front());
    ASSERT_TRUE(gradients);
    ASSERT_EQ(gradients.size(), 2u);
    EXPECT_TRUE(gradients.getType(0).isF32());
    EXPECT_EQ(gradients.getType(1), RankedTensorType::get({3}, Float32Type::get(&context)));
}

TEST_F(VernonStructuredVjpTest, RecordsAndReversesOnlyTheSelectedIfBranch) {
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
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"x"}, "structured_forward", "structured_backward"});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    EXPECT_EQ(result->forward.getOps<AdCaptureOp>().empty(), false);
    unsigned predicates = 0;
    result->forward.walk([&](AdWriteLeafOp write) {
        if (write.getValue().getType().isInteger(1))
            ++predicates;
    });
    EXPECT_EQ(predicates, 1u);
    unsigned reverseBranches = 0;
    result->backward.walk([&](scf::IfOp) { ++reverseBranches; });
    EXPECT_EQ(reverseBranches, 1u);
}

TEST_F(VernonStructuredVjpTest, SavesBranchLocalPrimalsInsideTheirDefiningBranches) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(%x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
                    %condition: i1 {vernon.source_name = "condition", vernon.abi_leaf_dtypes = ["bool"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) attributes {vernon.entry} {
    %result = scf.if %condition -> (f32) {
      %value = math.exp %x : f32
      scf.yield %value : f32
    } else {
      %value = math.sin %x : f32
      scf.yield %value : f32
    }
    func.return %result : f32
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result = buildStructuredVjp(
        module->lookupSymbol<func::FuncOp>("primal"), StructuredVjpOptions{{"x"}, "branch_forward", "branch_backward"});
    ASSERT_TRUE(succeeded(result));
    ASSERT_TRUE(succeeded(verify(*module)));

    scf::IfOp forwardIf;
    result->forward.walk([&](scf::IfOp candidate) { forwardIf = candidate; });
    ASSERT_TRUE(forwardIf);
    unsigned thenWrites = 0;
    forwardIf.getThenRegion().walk([&](AdWriteLeafOp) { ++thenWrites; });
    unsigned elseWrites = 0;
    forwardIf.getElseRegion().walk([&](AdWriteLeafOp) { ++elseWrites; });
    EXPECT_EQ(thenWrites, 1u);
    EXPECT_EQ(elseWrites, 0u);

    scf::IfOp backwardIf;
    result->backward.walk([&](scf::IfOp candidate) { backwardIf = candidate; });
    ASSERT_TRUE(backwardIf);
    unsigned thenReads = 0;
    backwardIf.getThenRegion().walk([&](AdReadLeafOp) { ++thenReads; });
    unsigned elseReads = 0;
    backwardIf.getElseRegion().walk([&](AdReadLeafOp) { ++elseReads; });
    EXPECT_EQ(thenReads, 1u);
    EXPECT_EQ(elseReads, 0u);
}

TEST_F(VernonStructuredVjpTest, AccumulatesPromotedMultiWrtAdjointsAcrossBranches) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f16 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f16"]},
      %y: f16 {vernon.source_name = "y", vernon.abi_leaf_dtypes = ["f16"]},
      %condition: i1 {vernon.source_name = "condition", vernon.abi_leaf_dtypes = ["bool"]})
      -> (f16 {vernon.abi_leaf_dtypes = ["f16"]}) attributes {vernon.entry} {
    %result = scf.if %condition -> (f16) {
      %product = arith.mulf %x, %y : f16
      scf.yield %product : f16
    } else {
      %sum = arith.addf %x, %y : f16
      scf.yield %sum : f16
    }
    func.return %result : f16
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"x", "y"}, "multi_forward", "multi_backward"});
    ASSERT_TRUE(succeeded(result));
    ASSERT_TRUE(succeeded(verify(*module)));
    auto gradients = dyn_cast<TupleType>(result->backward.getResultTypes().front());
    ASSERT_TRUE(gradients);
    ASSERT_EQ(gradients.size(), 2u);
    EXPECT_TRUE(llvm::all_of(gradients.getTypes(), [](Type type) { return type.isF32(); }));
    scf::IfOp reverseIf;
    result->backward.walk([&](scf::IfOp candidate) { reverseIf = candidate; });
    ASSERT_TRUE(reverseIf);
    EXPECT_EQ(reverseIf.getNumResults(), 2u);
    EXPECT_TRUE(llvm::all_of(reverseIf.getResultTypes(), [](Type type) { return type.isF32(); }));
}

TEST_F(VernonStructuredVjpTest, ClonesInactiveControlFlowAlongsideActiveRegions) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(%x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
                    %condition: i1 {vernon.source_name = "condition", vernon.abi_leaf_dtypes = ["bool"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) attributes {vernon.entry} {
    %unused = scf.if %condition -> (i32) {
      %one = arith.constant 1 : i32
      scf.yield %one : i32
    } else {
      %two = arith.constant 2 : i32
      scf.yield %two : i32
    }
    %result = scf.if %condition -> (f32) {
      %value = math.exp %x : f32
      scf.yield %value : f32
    } else {
      %value = math.sin %x : f32
      scf.yield %value : f32
    }
    func.return %result : f32
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result = buildStructuredVjp(
        module->lookupSymbol<func::FuncOp>("primal"), StructuredVjpOptions{{"x"}, "mixed_forward", "mixed_backward"});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    unsigned forwardBranches = 0;
    result->forward.walk([&](scf::IfOp) { ++forwardBranches; });
    EXPECT_EQ(forwardBranches, 2u);
}

TEST_F(VernonStructuredVjpTest, ReversesDataDependentWhileInDescendingRuntimeOrder) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]},
      %count: i32 {vernon.source_name = "count", vernon.abi_leaf_dtypes = ["i32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) attributes {vernon.entry} {
    %zero = arith.constant 0 : i32
    %loop:2 = scf.while (%value = %x, %index = %zero) : (f32, i32) -> (f32, i32) {
      %condition = arith.cmpi slt, %index, %count : i32
      scf.condition(%condition) %value, %index : f32, i32
    } do {
    ^bb0(%value: f32, %index: i32):
      %one = arith.constant 1 : i32
      %first = arith.cmpi eq, %index, %zero : i32
      %next = scf.if %first -> (f32) {
        %product = arith.mulf %value, %x : f32
        scf.yield %product : f32
      } else {
        %sum = arith.addf %value, %x : f32
        scf.yield %sum : f32
      }
      %next_index = arith.addi %index, %one : i32
      scf.yield %next, %next_index : f32, i32
    }
    func.return %loop#0 : f32
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result = buildStructuredVjp(
        module->lookupSymbol<func::FuncOp>("primal"), StructuredVjpOptions{{"x"}, "while_forward", "while_backward"});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    unsigned reservations = 0;
    unsigned checkedCounts = 0;
    result->forward.walk([&](AdReserveRecordOp) { ++reservations; });
    result->forward.walk([&](AdCheckedIncrementOp) { ++checkedCounts; });
    EXPECT_EQ(reservations, 3u);
    EXPECT_EQ(checkedCounts, 1u);
    unsigned reverseLoops = 0;
    result->backward.walk([&](scf::ForOp) { ++reverseLoops; });
    EXPECT_EQ(reverseLoops, 1u);
    unsigned nestedHandles = 0;
    result->backward.walk([&](AdReadNestedRegionOp) { ++nestedHandles; });
    EXPECT_EQ(nestedHandles, 2u);
    EXPECT_EQ(result->backward.getNumArguments(), 3u);
}

TEST_F(VernonStructuredVjpTest, RecordsBreakAndReturnExitKinds) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) attributes {vernon.entry} {
    %control0 = arith.constant 0 : i32
    %false = arith.constant false
    %true0 = arith.constant true
    %loop:3 = scf.while (%value = %x, %control = %control0, %returned = %false)
        : (f32, i32, i1) -> (f32, i32, i1) {
      %not_break = arith.cmpi eq, %control, %control0 : i32
      %not_returned = arith.xori %returned, %true0 : i1
      %active = arith.andi %not_break, %not_returned : i1
      scf.condition(%active) %value, %control, %returned : f32, i32, i1
    } do {
    ^bb0(%value: f32, %control: i32, %returned: i1):
      %zero = arith.constant 0.0 : f32
      %positive = arith.cmpf ogt, %value, %zero : f32
      %next:3 = scf.if %positive -> (f32, i32, i1) {
        %product = arith.mulf %value, %x : f32
        %true = arith.constant true
        scf.yield %product, %control, %true : f32, i32, i1
      } else {
        %sum = arith.addf %value, %x : f32
        %break = arith.constant 1 : i32
        scf.yield %sum, %break, %returned : f32, i32, i1
      }
      scf.yield %next#0, %next#1, %next#2 : f32, i32, i1
    } attributes {vernon.loop_control_index = 1 : i64, vernon.return_flag_index = 2 : i64}
    func.return %loop#0 : f32
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result = buildStructuredVjp(
        module->lookupSymbol<func::FuncOp>("primal"), StructuredVjpOptions{{"x"}, "exit_forward", "exit_backward"});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    AdEndRegionOp loopEnd;
    result->forward.walk([&](AdEndRegionOp end) {
        if (end.getExecutedCount().getDefiningOp<scf::WhileOp>())
            loopEnd = end;
    });
    ASSERT_TRUE(loopEnd);
    scf::WhileOp generatedLoop = loopEnd.getExecutedCount().getDefiningOp<scf::WhileOp>();
    auto returnSelect = loopEnd.getExitKind().getDefiningOp<arith::SelectOp>();
    ASSERT_TRUE(returnSelect);
    EXPECT_EQ(returnSelect.getCondition(), generatedLoop.getResult(2));
    auto breakSelect = returnSelect.getFalseValue().getDefiningOp<arith::SelectOp>();
    ASSERT_TRUE(breakSelect);
    auto breakCompare = breakSelect.getCondition().getDefiningOp<arith::CmpIOp>();
    ASSERT_TRUE(breakCompare);
    EXPECT_TRUE(breakCompare.getLhs() == generatedLoop.getResult(1) ||
                breakCompare.getRhs() == generatedLoop.getResult(1));
}

TEST_F(VernonStructuredVjpTest, DoesNotInferExitKindsFromConditionShape) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(%x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) attributes {vernon.entry} {
    %control = arith.constant 0 : i32
    %loop:2 = scf.while (%value = %x, %state = %control) : (f32, i32) -> (f32, i32) {
      %one = arith.constant 1 : i32
      %condition = arith.cmpi ne, %state, %one : i32
      scf.condition(%condition) %value, %state : f32, i32
    } do {
    ^bb0(%value: f32, %state: i32):
      %next = arith.mulf %value, %x : f32
      %one = arith.constant 1 : i32
      scf.yield %next, %one : f32, i32
    }
    func.return %loop#0 : f32
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result = buildStructuredVjp(
        module->lookupSymbol<func::FuncOp>("primal"), StructuredVjpOptions{{"x"}, "shape_forward", "shape_backward"});
    ASSERT_TRUE(succeeded(result));
    unsigned classifiedLoopExits = 0;
    result->forward.walk([&](AdEndRegionOp end) {
        if (end.getExecutedCount().getDefiningOp<scf::WhileOp>() && end.getExitKind().getDefiningOp<arith::SelectOp>())
            ++classifiedLoopExits;
    });
    EXPECT_EQ(classifiedLoopExits, 0u);
}

TEST_F(VernonStructuredVjpTest, RejectsMalformedLoopStateMetadataWithoutMutation) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(%x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) attributes {vernon.entry} {
    %loop = scf.while (%value = %x) : (f32) -> (f32) {
      %condition = arith.constant true
      scf.condition(%condition) %value : f32
    } do {
    ^bb0(%value: f32):
      %next = arith.mulf %value, %x : f32
      scf.yield %next : f32
    } attributes {vernon.loop_control_index = 0 : i64}
    func.return %loop : f32
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    func::FuncOp primal = module->lookupSymbol<func::FuncOp>("primal");
    EXPECT_TRUE(failed(buildStructuredVjp(primal, StructuredVjpOptions{{"x"}, "invalid_forward", "invalid_backward"})));
    EXPECT_TRUE(primal->hasAttr("vernon.entry"));
    EXPECT_FALSE(module->lookupSymbol<func::FuncOp>("invalid_forward"));
    EXPECT_FALSE(module->lookupSymbol<func::FuncOp>("invalid_backward"));
}

TEST_F(VernonStructuredVjpTest, HasNoCompileTimeTripCountCap) {
    for (int32_t limit : {0, 1, 2048}) {
        std::string source = (Twine(R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.abi_leaf_dtypes = ["f32"]})
      -> (f32 {vernon.abi_leaf_dtypes = ["f32"]}) attributes {vernon.entry} {
    %zero = arith.constant 0 : i32
    %limit = arith.constant )mlir") +
                              Twine(limit) + R"mlir( : i32
    %loop:2 = scf.while (%value = %x, %index = %zero) : (f32, i32) -> (f32, i32) {
      %condition = arith.cmpi slt, %index, %limit : i32
      scf.condition(%condition) %value, %index : f32, i32
    } do {
    ^bb0(%value: f32, %index: i32):
      %next = arith.addf %value, %x : f32
      %one = arith.constant 1 : i32
      %next_index = arith.addi %index, %one : i32
      scf.yield %next, %next_index : f32, i32
    }
    func.return %loop#0 : f32
  }
}
)mlir")
                                 .str();
        OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, ParserConfig(&context));
        ASSERT_TRUE(module) << limit;
        std::string suffix = std::to_string(limit);
        FailureOr<StructuredVjpResult> result =
            buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                               StructuredVjpOptions{{"x"}, "limit_forward_" + suffix, "limit_backward_" + suffix});
        ASSERT_TRUE(succeeded(result)) << limit;
        EXPECT_TRUE(succeeded(verify(*module))) << limit;
        unsigned reverseLoops = 0;
        result->backward.walk([&](scf::ForOp) { ++reverseLoops; });
        EXPECT_EQ(reverseLoops, 1u) << limit;
    }
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
    EXPECT_TRUE(
        failed(buildStructuredVjp(primal, StructuredVjpOptions{{"x"}, "structured_forward", "structured_backward"})));
    EXPECT_TRUE(primal->hasAttr("vernon.entry"));
    EXPECT_FALSE(module->lookupSymbol<func::FuncOp>("structured_forward"));
    EXPECT_FALSE(module->lookupSymbol<func::FuncOp>("structured_backward"));
}

} // namespace
} // namespace mlir::vernon
