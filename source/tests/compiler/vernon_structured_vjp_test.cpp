#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vernon/IR/Vernon.h"
#include "mlir/Dialect/Vernon/IR/VernonValueAbi.h"
#include "mlir/Dialect/Vernon/Transforms/VernonGPUProfileABI.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerAccumulation.h"
#include "mlir/Dialect/Vernon/Transforms/VernonLowerGPUAutodiff.h"
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

TEST_F(VernonStructuredVjpTest, RankZeroTensorViewModelsScalarStorage) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @value(%tensor: !vernon.tensor<f32, []>) {
    func.return
  }
  func.func @accumulate(
      %gradient: !vernon.tensor_view<f32, [], "read_write", "device">) {
    %value = arith.constant 1.0 : f32
    "vernon.reduce_sum"(%value, %gradient) {deterministic = false}
        : (f32, !vernon.tensor_view<f32, [], "read_write", "device">) -> ()
    func.return
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    EXPECT_TRUE(succeeded(verify(*module)));
}

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
        const bool noTape = result->tapeBytes == 0;
        EXPECT_EQ(result->forward.getNumResults(), noTape ? 0u : 1u);
        EXPECT_EQ(result->backward.getNumResults(), 1u);
        StringRef storageKind = noTape ? "none" : "static";
        EXPECT_EQ(result->forward->getAttrOfType<StringAttr>("vernon.ad.residual_storage").getValue(), storageKind);
        EXPECT_EQ(result->backward->getAttrOfType<StringAttr>("vernon.ad.residual_storage").getValue(), storageKind);
        ASSERT_EQ(result->derivativeRules.size(), 1u);
        EXPECT_EQ(result->derivativeRules.front(), testCase.first.str());
        unsigned backwardBarriers = 0;
        result->backward.walk([&](BarrierOp) { ++backwardBarriers; });
        EXPECT_EQ(backwardBarriers, 0u);
    }
}

TEST_F(VernonStructuredVjpTest, NoTapeProfileUsesExplicitPrimalArguments) {
    OwningOpRef<ModuleOp> module = parseStorageObjective("arith.mulf");
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"x", "y"}, "static_forward", "static_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));

    SmallVector<int64_t> writeOffsets;
    SmallVector<int64_t> readOffsets;
    result->forward.walk([&](AdWriteLeafOp operation) {
        writeOffsets.push_back(operation->getAttrOfType<IntegerAttr>("leaf_offset").getInt());
    });
    result->backward.walk([&](AdReadLeafOp operation) {
        readOffsets.push_back(operation->getAttrOfType<IntegerAttr>("leaf_offset").getInt());
    });
    llvm::sort(writeOffsets);
    llvm::sort(readOffsets);
    EXPECT_TRUE(writeOffsets.empty());
    EXPECT_EQ(writeOffsets, readOffsets);

    unsigned forwardRegions = 0;
    result->forward.walk([&](AdBeginRegionOp) { ++forwardRegions; });
    EXPECT_EQ(forwardRegions, 0u);
    EXPECT_EQ(result->forward->getAttrOfType<StringAttr>("vernon.ad.residual_storage").getValue(), "none");
    EXPECT_EQ(result->tapeBytes, 0u);
    EXPECT_EQ(result->requiredPrimalPaths, (SmallVector<std::string>{"primal.x", "primal.y"}));
    EXPECT_GT(result->backward->getAttrOfType<IntegerAttr>("vernon.ad.active_operation_count").getInt(), 0);
    EXPECT_GE(result->backward->getAttrOfType<IntegerAttr>("vernon.ad.recomputation_cost").getInt(), 0);
    unsigned captures = 0;
    result->forward.walk([&](AdCaptureOp) { ++captures; });
    EXPECT_EQ(captures, 0u);
    EXPECT_TRUE(llvm::none_of(result->forward.getArgumentTypes(), [](Type type) { return isa<AdTapeType>(type); }));
    EXPECT_TRUE(llvm::none_of(result->forward.getResultTypes(), [](Type type) { return isa<AdTapeType>(type); }));
    EXPECT_TRUE(llvm::none_of(result->backward.getArgumentTypes(), [](Type type) { return isa<AdTapeType>(type); }));

    const unsigned forwardArguments = result->forward.getNumArguments();
    const unsigned backwardArguments = result->backward.getNumArguments();
    result->backward->setAttr("vernon.workgroup_size", DenseI32ArrayAttr::get(&context, {1, 1, 1}));
    PassManager manager(&context);
    manager.addPass(createVernonLowerGPUAutodiffPass());
    ASSERT_TRUE(succeeded(manager.run(*module)));
    EXPECT_EQ(result->forward.getNumArguments(), forwardArguments);
    EXPECT_EQ(result->backward.getNumArguments(), backwardArguments + 4);
    EXPECT_EQ(result->backward.getNumResults(), 0u);
    EXPECT_EQ(
        std::distance(result->backward.getOps<ReduceSumOp>().begin(), result->backward.getOps<ReduceSumOp>().end()), 2);
    unsigned launchMetadata = 0;
    unsigned physicalGlobalIds = 0;
    for (unsigned index = backwardArguments + 2; index < result->backward.getNumArguments(); ++index) {
        if (auto role = result->backward.getArgAttrOfType<StringAttr>(index, "vernon.autodiff_role");
            role && role.getValue() == "launch_metadata")
            ++launchMetadata;
        if (auto builtin = result->backward.getArgAttrOfType<StringAttr>(index, "vernon.builtin");
            builtin && builtin.getValue() == "global_invocation_id" &&
            isa<RankedTensorType>(result->backward.getArgument(index).getType()))
            ++physicalGlobalIds;
    }
    EXPECT_EQ(launchMetadata, 1u);
    EXPECT_EQ(physicalGlobalIds, 1u);
    EXPECT_TRUE(llvm::none_of(result->forward.getArguments(), [&](BlockArgument argument) {
        auto source = result->forward.getArgAttrOfType<StringAttr>(argument.getArgNumber(), "vernon.source_name");
        return source && source.getValue().starts_with("__vernon_ad_");
    }));

    const unsigned loweredBackwardArguments = result->backward.getNumArguments();
    SmallVector<Type> loweredBackwardTypes(result->backward.getArgumentTypes());
    PassManager secondManager(&context);
    secondManager.addPass(createVernonLowerGPUAutodiffPass());
    ASSERT_TRUE(succeeded(secondManager.run(*module)));
    EXPECT_EQ(result->backward.getNumArguments(), loweredBackwardArguments);
    EXPECT_EQ(result->backward.getArgumentTypes(), ArrayRef<Type>(loweredBackwardTypes));
}

TEST_F(VernonStructuredVjpTest, GPUProfileBindingsAreMaterializedAfterLogicalVjp) {
    OwningOpRef<ModuleOp> module = parseStorageObjective("arith.mulf");
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"x", "y"}, "gpu_forward", "gpu_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));

    SmallVector<unsigned> viewArguments;
    for (auto [index, type] : llvm::enumerate(result->backward.getArgumentTypes())) {
        if (!isa<TensorViewType>(type))
            continue;
        viewArguments.push_back(index);
        DictionaryAttr attrs = result->backward.getArgAttrDict(index);
        EXPECT_EQ(attrs.getAs<StringAttr>("vernon.interface").getValue(), "input");
        EXPECT_TRUE(attrs.get("vernon.location"));
        EXPECT_FALSE(attrs.get("vernon.set"));
        EXPECT_FALSE(attrs.get("vernon.binding"));
        if (attrs.getAs<StringAttr>("vernon.source_name").getValue() == "loss")
            EXPECT_EQ(attrs.getAs<StringAttr>("vernon.autodiff_role").getValue(), "cotangent");
    }
    ASSERT_FALSE(viewArguments.empty());

    (*module)->setAttr("vernon.ad_profile", StringAttr::get(&context, "backward"));
    ASSERT_TRUE(succeeded(materializeGPUAutodiffProfileBindings(*module)));
    for (unsigned index : viewArguments) {
        DictionaryAttr attrs = result->backward.getArgAttrDict(index);
        EXPECT_EQ(attrs.getAs<StringAttr>("vernon.interface").getValue(), "resource");
        EXPECT_FALSE(attrs.get("vernon.location"));
        EXPECT_EQ(attrs.getAs<IntegerAttr>("vernon.set").getInt(), 0);
        EXPECT_EQ(attrs.getAs<IntegerAttr>("vernon.binding").getInt(), index);
    }
}

TEST_F(VernonStructuredVjpTest, StaticCaptureOffsetsMatchWithinCurrentProfileContract) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: index {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute", vernon.ad.planning_policy = "min_runtime"} {
    %square = arith.mulf %x, %x : f32
    %fourth = arith.mulf %square, %square : f32
    %result = arith.divf %fourth, %x : f32
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
                           StructuredVjpOptions{{"x"}, "static_forward", "static_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    EXPECT_GT(result->tapeBytes, 0u);
    EXPECT_EQ(result->forward->getAttrOfType<StringAttr>("vernon.ad.residual_storage").getValue(), "static");
    SmallVector<int64_t> writeOffsets;
    SmallVector<int64_t> readOffsets;
    result->forward.walk([&](AdWriteLeafOp operation) {
        writeOffsets.push_back(operation->getAttrOfType<IntegerAttr>("leaf_offset").getInt());
    });
    result->backward.walk([&](AdReadLeafOp operation) {
        readOffsets.push_back(operation->getAttrOfType<IntegerAttr>("leaf_offset").getInt());
    });
    llvm::sort(writeOffsets);
    llvm::sort(readOffsets);
    EXPECT_FALSE(writeOffsets.empty());
    EXPECT_EQ(writeOffsets, readOffsets);

    DenseI32ArrayAttr workgroup = DenseI32ArrayAttr::get(&context, {8, 1, 1});
    result->forward->setAttr("vernon.workgroup_size", workgroup);
    result->backward->setAttr("vernon.workgroup_size", workgroup);
    (*module)->setAttr("vernon.ad_profile", StringAttr::get(&context, "static"));
    ASSERT_TRUE(succeeded(materializeGPUAutodiffProfileBindings(*module)));
    PassManager manager(&context);
    manager.addPass(createVernonLowerGPUAutodiffPass());
    ASSERT_TRUE(succeeded(manager.run(*module)));
    ASSERT_TRUE(succeeded(verify(*module)));
    EXPECT_EQ(result->backward.getNumResults(), 0u);
    unsigned gradientResources = 0;
    unsigned carriedCotangents = 0;
    for (unsigned index = 0; index < result->backward.getNumArguments(); ++index) {
        auto role = result->backward.getArgAttrOfType<StringAttr>(index, "vernon.autodiff_role");
        auto view = dyn_cast<TensorViewType>(result->backward.getArgument(index).getType());
        if (!role || !view)
            continue;
        if (role.getValue() == "gradient") {
            ++gradientResources;
            EXPECT_TRUE(view.getShape().empty());
            EXPECT_EQ(result->backward.getArgAttrOfType<StringAttr>(index, "vernon.autodiff_source").getValue(), "x");
        } else if (role.getValue() == "cotangent") {
            ++carriedCotangents;
            EXPECT_EQ(view.getShape(), (ArrayRef<int64_t>{-1, 1}));
        }
    }
    EXPECT_EQ(gradientResources, 1u);
    EXPECT_EQ(carriedCotangents, 1u);
    EXPECT_EQ(
        std::distance(result->backward.getOps<ReduceSumOp>().begin(), result->backward.getOps<ReduceSumOp>().end()), 1);
    for (LoadOp load : result->backward.getOps<LoadOp>())
        if (auto role = dyn_cast<BlockArgument>(load.getStorage())
                            ? result->backward.getArgAttrOfType<StringAttr>(
                                  cast<BlockArgument>(load.getStorage()).getArgNumber(), "vernon.autodiff_role")
                            : StringAttr{};
            role && role.getValue() == "cotangent")
            EXPECT_EQ(load.getIndices().size(), 2u);
    for (func::FuncOp function : {result->forward, result->backward}) {
        EXPECT_TRUE(llvm::none_of(function.getArgumentTypes(), containsLogicalAutodiffHandle));
        bool hasLogicalTape = false;
        function.walk([&](Operation *operation) {
            hasLogicalTape =
                hasLogicalTape ||
                isa<AdCaptureOp, AdBeginRegionOp, AdReserveRecordOp, AdReadLeafOp, AdReadNestedRegionOp>(operation);
        });
        EXPECT_FALSE(hasLogicalTape);
    }
}

TEST_F(VernonStructuredVjpTest, LowersCapturedProfilesToReflectedGpuTapeResources) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: index {vernon.builtin = "global_invocation_id"})
      attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>,
        vernon.ad.planning_policy = "min_runtime"
      } {
    %square = arith.mulf %x, %x : f32
    %fourth = arith.mulf %square, %square : f32
    %result = arith.divf %fourth, %x : f32
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
                           StructuredVjpOptions{{"x"}, "gpu_tape_forward", "gpu_tape_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    ASSERT_GT(result->tapeBytes, 0u);
    (*module)->setAttr("vernon.ad_profile", StringAttr::get(&context, "captured"));
    ASSERT_TRUE(succeeded(materializeGPUAutodiffProfileBindings(*module)));

    PassManager manager(&context);
    manager.addPass(createVernonLowerGPUAutodiffPass());
    ASSERT_TRUE(succeeded(manager.run(*module)));
    ASSERT_TRUE(succeeded(verify(*module)));

    for (StringRef name : {"gpu_tape_forward", "gpu_tape_backward"}) {
        func::FuncOp function = module->lookupSymbol<func::FuncOp>(name);
        ASSERT_TRUE(function);
        unsigned tapeResources = 0;
        unsigned segmentResources = 0;
        for (unsigned index = 0; index < function.getNumArguments(); ++index) {
            auto resource = function.getArgAttrOfType<StringAttr>(index, "vernon.source_name");
            if (!resource || !resource.getValue().starts_with("__vernon_ad_"))
                continue;
            EXPECT_TRUE(isa<TensorViewType>(function.getArgumentTypes()[index]));
            StringRef expectedRole = resource.getValue() == "__vernon_ad_tape"      ? "tape"
                                     : resource.getValue() == "__vernon_ad_segment" ? "replay_segment"
                                                                                    : "replay_status";
            EXPECT_EQ(function.getArgAttrOfType<StringAttr>(index, "vernon.autodiff_role").getValue(), expectedRole);
            tapeResources += resource.getValue() == "__vernon_ad_tape";
            segmentResources += resource.getValue() == "__vernon_ad_segment";
        }
        EXPECT_EQ(tapeResources, 1u);
        EXPECT_EQ(segmentResources, 1u);
        EXPECT_TRUE(llvm::none_of(function.getArgumentTypes(), containsLogicalAutodiffHandle));
        bool hasLogicalOperation = false;
        function.walk([&](Operation *operation) {
            hasLogicalOperation =
                hasLogicalOperation || isa<AdCaptureOp, AdBeginInvocationOp, AdBeginRegionOp, AdReserveRecordOp,
                                           AdWriteLeafOp, AdReadLeafOp, AdReadNestedRegionOp>(operation);
        });
        EXPECT_FALSE(hasLogicalOperation);
        unsigned physicalLoads = 0;
        function.walk([&](PhysicalLoadOp) { ++physicalLoads; });
        EXPECT_GT(physicalLoads, 0u);
    }
}

TEST_F(VernonStructuredVjpTest, RematerializedStraightLineValueHasNoTapeTraffic) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: index {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %gid_i32 = arith.index_cast %gid : index to i32
    %coefficient = arith.sitofp %gid_i32 : i32 to f32
    %result = arith.mulf %x, %coefficient : f32
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
                           StructuredVjpOptions{{"x"}, "remat_forward", "remat_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    unsigned writes = 0;
    unsigned reads = 0;
    result->forward.walk([&](AdWriteLeafOp) { ++writes; });
    result->backward.walk([&](AdReadLeafOp) { ++reads; });
    EXPECT_EQ(writes, 0u);
    EXPECT_EQ(reads, 0u);
    EXPECT_GT(result->backward->getAttrOfType<IntegerAttr>("vernon.ad.recomputation_cost").getInt(), 0);
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

TEST_F(VernonStructuredVjpTest, ScalarizesSharedTensorViewGradientLeaves) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %values: !vernon.tensor_view<tensor<2xf32>, [1], "read", "device">
          {vernon.source_name = "values", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %index: index {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %value = "vernon.load"(%values, %index)
        : (!vernon.tensor_view<tensor<2xf32>, [1], "read", "device">, index) -> tensor<2xf32>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %left = tensor.extract %value[%zero] : tensor<2xf32>
    %right = tensor.extract %value[%one] : tensor<2xf32>
    %sum = arith.addf %left, %right : f32
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
                           StructuredVjpOptions{{"values"}, "tensor_forward", "tensor_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));

    unsigned scalarScatterAdds = 0;
    result->backward.walk([&](ScatterAddOp scatter) {
        EXPECT_TRUE(scatter.getValue().getType().isF32());
        EXPECT_EQ(scatter.getIndices().size(), 2u);
        ++scalarScatterAdds;
    });
    EXPECT_EQ(scalarScatterAdds, 2u);

    const unsigned gradientArgument = result->backward.getNumArguments() - 1;
    auto gradient = dyn_cast<TensorViewType>(result->backward.getArgument(gradientArgument).getType());
    ASSERT_TRUE(gradient);
    EXPECT_TRUE(gradient.getElementType().isF32());
    EXPECT_EQ(gradient.getShape(), ArrayRef<int64_t>({1, 2}));
    auto ownership = result->backward.getArgAttrOfType<StringAttr>(gradientArgument, kAccumulationOwnershipAttrName);
    ASSERT_TRUE(ownership);
    EXPECT_EQ(ownership.getValue(), "atomic_shared");
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
    EXPECT_EQ(result->forward->getAttrOfType<StringAttr>("vernon.ad.residual_storage").getValue(), "none");
    EXPECT_EQ(result->tapeBytes, 0u);
    EXPECT_EQ(result->requiredPrimalPaths, (SmallVector<std::string>{"primal.values"}));
    ASSERT_EQ(result->backward.getNumArguments(), 4u);
    auto gradient = dyn_cast<TensorViewType>(result->backward.getArgumentTypes().back());
    ASSERT_TRUE(gradient);
    EXPECT_EQ(gradient.getShape(), ArrayRef<int64_t>({-1}));
    EXPECT_EQ(gradient.getAccess(), "write");
    unsigned reverseReloads = 0;
    result->backward.walk([&](LoadOp) { ++reverseReloads; });
    EXPECT_GT(reverseReloads, 0u);
    unsigned dynamicBuffers = 0;
    result->backward.walk([&](AdAdjointBufferCreateOp create) {
        dynamicBuffers += llvm::is_contained(create.getBuffer().getType().getShape(), int64_t{-1});
    });
    EXPECT_EQ(dynamicBuffers, 0u);
}

TEST_F(VernonStructuredVjpTest, ReconstructsCanonicalForWithoutLoopHistory) {
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
    EXPECT_EQ(result->tapeBytes, 0u);
    EXPECT_EQ(result->requiredPrimalPaths, (SmallVector<std::string>{"primal.count"}));
    EXPECT_EQ(result->forward->getAttrOfType<StringAttr>("vernon.ad.residual_storage").getValue(), "none");
    EXPECT_EQ(result->backward->getAttrOfType<StringAttr>("vernon.ad.residual_storage").getValue(), "none");
    unsigned dynamicRegions = 0;
    result->forward.walk([&](AdBeginRegionOp) { ++dynamicRegions; });
    EXPECT_EQ(dynamicRegions, 0u);
    unsigned forwardFors = 0;
    result->forward.walk([&](scf::ForOp) { ++forwardFors; });
    EXPECT_EQ(forwardFors, 1u);
    unsigned backwardFors = 0;
    result->backward.walk([&](scf::ForOp) { ++backwardFors; });
    EXPECT_EQ(backwardFors, 1u);
    unsigned executedCountReads = 0;
    result->backward.walk([&](AdReadExecutedCountOp) { ++executedCountReads; });
    EXPECT_EQ(executedCountReads, 0u);
    unsigned sourceFors = 0;
    module->lookupSymbol<func::FuncOp>("primal").walk([&](scf::ForOp) { ++sourceFors; });
    EXPECT_EQ(sourceFors, 1u);
    EXPECT_EQ(llvm::range_size(module->getOps<func::FuncOp>()), 3u);
}

TEST_F(VernonStructuredVjpTest, CapturesOnlyRequiredLoopCarriedPrimalPerIteration) {
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
      %next = arith.mulf %value, %x : f32
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
                           StructuredVjpOptions{{"x"}, "for_capture_forward", "for_capture_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    EXPECT_GT(result->tapeBytes, 0u);
    EXPECT_EQ(result->forward->getAttrOfType<StringAttr>("vernon.ad.residual_storage").getValue(), "dynamic");
    unsigned forwardFors = 0;
    result->forward.walk([&](scf::ForOp) { ++forwardFors; });
    EXPECT_EQ(forwardFors, 1u);
    unsigned backwardFors = 0;
    result->backward.walk([&](scf::ForOp) { ++backwardFors; });
    EXPECT_EQ(backwardFors, 1u);
    unsigned predicateOrCountReads = 0;
    result->backward.walk([&](AdReadExecutedCountOp) { ++predicateOrCountReads; });
    EXPECT_EQ(predicateOrCountReads, 0u);
    unsigned primalReads = 0;
    result->backward.walk([&](AdReadLeafOp) { ++primalReads; });
    EXPECT_GT(primalReads, 0u);
}

TEST_F(VernonStructuredVjpTest, ReconstructsNestedForWithExactVersionReloads) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %input: !vernon.tensor_view<f32, [2, 2], "read", "device">
          {vernon.source_name = "input", vernon.abi_leaf_dtypes = ["f32"]},
      %target: !vernon.tensor_view<f32, [2, 2], "read", "device">
          {vernon.source_name = "target", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %zero_value = arith.constant 0.0 : f32
    %result = scf.for %y = %zero to %two step %one
        iter_args(%row_accumulator = %zero_value) -> (f32) {
      %row_result = scf.for %x = %zero to %two step %one
          iter_args(%accumulator = %row_accumulator) -> (f32) {
        %input_value = "vernon.load"(%input, %y, %x)
            : (!vernon.tensor_view<f32, [2, 2], "read", "device">, index, index) -> f32
        %target_value = "vernon.load"(%target, %y, %x)
            : (!vernon.tensor_view<f32, [2, 2], "read", "device">, index, index) -> f32
        %difference = arith.subf %input_value, %target_value : f32
        %squared = arith.mulf %difference, %difference : f32
        %next = arith.addf %accumulator, %squared : f32
        scf.yield %next : f32
      }
      scf.yield %row_result : f32
    }
    "vernon.store"(%result, %loss, %zero)
        : (f32, !vernon.tensor_view<f32, [1], "write", "device">, index) -> ()
    func.return
  }
}
)mlir",
        ParserConfig(&context));
    ASSERT_TRUE(module);
    FailureOr<StructuredVjpResult> result =
        buildStructuredVjp(module->lookupSymbol<func::FuncOp>("primal"),
                           StructuredVjpOptions{{"input"}, "nested_for_forward", "nested_for_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    EXPECT_EQ(result->tapeBytes, 0u);
    EXPECT_EQ(result->requiredPrimalPaths, (SmallVector<std::string>{"primal.input", "primal.target"}));
    EXPECT_EQ(result->forward->getAttrOfType<StringAttr>("vernon.ad.residual_storage").getValue(), "none");
    unsigned backwardFors = 0;
    result->backward.walk([&](scf::ForOp) { ++backwardFors; });
    EXPECT_EQ(backwardFors, 2u);
    unsigned exactReloads = 0;
    result->backward.walk([&](LoadOp) { ++exactReloads; });
    EXPECT_GE(exactReloads, 2u);
}

TEST_F(VernonStructuredVjpTest, ReconstructsPureIfWithoutControlHistory) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: index {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %zero = arith.constant 0.0 : f32
    %positive = arith.cmpf ogt, %x, %zero : f32
    %result = scf.if %positive -> (f32) {
      %squared = arith.mulf %x, %x : f32
      scf.yield %squared : f32
    } else {
      %negated = arith.negf %x : f32
      scf.yield %negated : f32
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
                           StructuredVjpOptions{{"x"}, "if_forward", "if_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    EXPECT_EQ(result->forward->getAttrOfType<StringAttr>("vernon.ad.residual_storage").getValue(), "none");
    EXPECT_EQ(result->backward->getAttrOfType<StringAttr>("vernon.ad.residual_storage").getValue(), "none");
    EXPECT_EQ(result->tapeBytes, 0u);
    EXPECT_EQ(result->requiredPrimalPaths, (SmallVector<std::string>{"primal.x"}));
    unsigned encodedRegions = 0;
    result->forward.walk([&](AdBeginRegionOp) { ++encodedRegions; });
    EXPECT_EQ(encodedRegions, 0u);
    unsigned backwardIfs = 0;
    result->backward.walk([&](scf::IfOp) { ++backwardIfs; });
    EXPECT_GT(backwardIfs, 0u);
    unsigned predicateReads = 0;
    result->backward.walk([&](AdReadLeafOp) { ++predicateReads; });
    EXPECT_EQ(predicateReads, 0u);
}

TEST_F(VernonStructuredVjpTest, ReconstructsIfPredicateFromCanonicalForInduction) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: index {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %two = arith.constant 2 : index
    %result = scf.for %index = %zero to %four step %one
        iter_args(%value = %x) -> (f32) {
      %first_half = arith.cmpi slt, %index, %two : index
      %next = scf.if %first_half -> (f32) {
        %sum = arith.addf %value, %x : f32
        scf.yield %sum : f32
      } else {
        %difference = arith.subf %value, %x : f32
        scf.yield %difference : f32
      }
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
                           StructuredVjpOptions{{"x"}, "nested_for_if_forward", "nested_for_if_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    EXPECT_EQ(result->tapeBytes, 0u);
    unsigned forwardRegions = 0;
    result->forward.walk([&](AdBeginRegionOp) { ++forwardRegions; });
    EXPECT_EQ(forwardRegions, 0u);
    unsigned backwardFors = 0;
    unsigned backwardIfs = 0;
    result->backward.walk([&](scf::ForOp) { ++backwardFors; });
    result->backward.walk([&](scf::IfOp) { ++backwardIfs; });
    EXPECT_EQ(backwardFors, 1u);
    EXPECT_EQ(backwardIfs, 1u);
}

TEST_F(VernonStructuredVjpTest, ReconstructsNestedIfInsideCapturedOuterControl) {
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        R"mlir(
module {
  func.func @primal(
      %x: f32 {vernon.source_name = "x", vernon.dtype = "f32", vernon.abi_leaf_dtypes = ["f32"]},
      %control: !vernon.tensor_view<f32, [1], "read_write", "device">
          {vernon.source_name = "control", vernon.abi_leaf_dtypes = ["f32"]},
      %loss: !vernon.tensor_view<f32, [1], "write", "device">
          {vernon.source_name = "loss", vernon.abi_leaf_dtypes = ["f32"]},
      %gid: index {vernon.builtin = "global_invocation_id"})
      attributes {vernon.entry, vernon.stage = "compute"} {
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %control_value = "vernon.load"(%control, %gid)
        : (!vernon.tensor_view<f32, [1], "read_write", "device">, index) -> f32
    %outer_condition = arith.cmpf ogt, %control_value, %zero : f32
    %result = scf.if %outer_condition -> (f32) {
      %inner_condition = arith.cmpf olt, %x, %one : f32
      %inner = scf.if %inner_condition -> (f32) {
        %squared = arith.mulf %x, %x : f32
        scf.yield %squared : f32
      } else {
        %negated = arith.negf %x : f32
        scf.yield %negated : f32
      }
      scf.yield %inner : f32
    } else {
      scf.yield %x : f32
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
                           StructuredVjpOptions{{"x"}, "nested_if_forward", "nested_if_backward", {"loss"}});
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verify(*module)));
    EXPECT_EQ(result->forward->getAttrOfType<StringAttr>("vernon.ad.residual_storage").getValue(), "dynamic");
    unsigned forwardRegions = 0;
    result->forward.walk([&](AdBeginRegionOp) { ++forwardRegions; });
    EXPECT_EQ(forwardRegions, 2u);
    unsigned backwardIfs = 0;
    result->backward.walk([&](scf::IfOp) { ++backwardIfs; });
    EXPECT_EQ(backwardIfs, 2u);
    unsigned predicateReads = 0;
    result->backward.walk([&](AdReadLeafOp) { ++predicateReads; });
    EXPECT_EQ(predicateReads, 1u);
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
      %next = arith.mulf %value, %x : f32
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
    EXPECT_EQ(result->forward->getAttrOfType<StringAttr>("vernon.ad.residual_storage").getValue(), "dynamic");
    EXPECT_EQ(result->backward->getAttrOfType<StringAttr>("vernon.ad.residual_storage").getValue(), "dynamic");
    unsigned dynamicRegions = 0;
    result->forward.walk([&](AdBeginRegionOp) { ++dynamicRegions; });
    EXPECT_GT(dynamicRegions, 1u);
    unsigned executedCountReads = 0;
    unsigned carriedPrimalReads = 0;
    result->backward.walk([&](AdReadExecutedCountOp) { ++executedCountReads; });
    result->backward.walk([&](AdReadLeafOp) { ++carriedPrimalReads; });
    EXPECT_EQ(executedCountReads, 1u);
    EXPECT_GT(carriedPrimalReads, 0u);
    SmallVector<int64_t> writeOffsets;
    SmallVector<int64_t> readOffsets;
    result->forward.walk([&](AdWriteLeafOp operation) {
        writeOffsets.push_back(operation->getAttrOfType<IntegerAttr>("leaf_offset").getInt());
    });
    result->backward.walk([&](AdReadLeafOp operation) {
        readOffsets.push_back(operation->getAttrOfType<IntegerAttr>("leaf_offset").getInt());
    });
    llvm::sort(writeOffsets);
    llvm::sort(readOffsets);
    EXPECT_FALSE(writeOffsets.empty());
    EXPECT_EQ(writeOffsets, readOffsets);

    DenseI32ArrayAttr workgroup = DenseI32ArrayAttr::get(&context, {8, 1, 1});
    result->forward->setAttr("vernon.workgroup_size", workgroup);
    result->backward->setAttr("vernon.workgroup_size", workgroup);
    (*module)->setAttr("vernon.ad_profile", StringAttr::get(&context, "dynamic"));
    ASSERT_TRUE(succeeded(materializeGPUAutodiffProfileBindings(*module)));
    PassManager manager(&context);
    manager.addPass(createVernonLowerGPUAutodiffPass());
    ASSERT_TRUE(succeeded(manager.run(*module)));
    ASSERT_TRUE(succeeded(verify(*module)));
    for (func::FuncOp function : {result->forward, result->backward}) {
        EXPECT_TRUE(llvm::none_of(function.getArgumentTypes(), containsLogicalAutodiffHandle));
        bool hasLogicalTape = false;
        function.walk([&](Operation *operation) {
            hasLogicalTape =
                hasLogicalTape ||
                isa<AdCaptureOp, AdBeginRegionOp, AdReserveRecordOp, AdReadLeafOp, AdReadNestedRegionOp>(operation);
        });
        EXPECT_FALSE(hasLogicalTape);
    }
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

TEST_F(VernonStructuredVjpTest, AccumulationPolicyUsesMeasuredContentionCrossover) {
    const auto run = [&](int32_t workgroupSize, AtomicAddImplementation implementation) {
        std::string source = (Twine(R"mlir(module {
  func.func @accumulate(
      %value: f32,
      %storage: !vernon.tensor_view<f32, [1], "read_write", "device">)
      attributes {vernon.entry, vernon.stage = "compute", vernon.workgroup_size = array<i32: )mlir") +
                              Twine(workgroupSize) + R"mlir(, 1, 1>} {
    %zero = arith.constant 0 : index
    "vernon.reduce_sum"(%value, %storage, %zero) {deterministic = false}
        : (f32, !vernon.tensor_view<f32, [1], "read_write", "device">, index) -> ()
    return
  }
})mlir")
                                 .str();
        OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, ParserConfig(&context));
        EXPECT_TRUE(module);
        if (!module)
            return std::pair<unsigned, unsigned>{0, 0};
        AccumulationTargetCapabilities capabilities;
        capabilities.device.f32 = implementation;
        capabilities.supportsWorkgroupReduction = true;
        capabilities.costModel.nativeAtomicReductionCrossover = 64;
        capabilities.costModel.integerCasReductionCrossover = 64;
        PassManager manager(&context);
        manager.addPass(createVernonLowerAccumulationPass(capabilities));
        EXPECT_TRUE(succeeded(manager.run(*module)));
        unsigned reductions = 0;
        unsigned atomics = 0;
        module->walk([&](Operation *operation) {
            if (auto reduce = dyn_cast<ReduceSumOp>(operation)) {
                auto strategy = reduce->getAttrOfType<StringAttr>(kAccumulationStrategyAttrName);
                reductions += strategy && strategy.getValue() == kWorkgroupReductionAccumulationStrategy;
            }
            if (isa<AtomicOp, PhysicalAtomicOp>(operation)) {
                auto strategy = operation->getAttrOfType<StringAttr>(kAccumulationStrategyAttrName);
                auto selected = operation->getAttrOfType<StringAttr>(kAtomicImplementationAttrName);
                StringRef expected = implementation == AtomicAddImplementation::Native
                                         ? kNativeAtomicImplementation
                                         : kIntegerCasAtomicImplementation;
                atomics += strategy && strategy.getValue() == kAtomicAccumulationStrategy && selected &&
                           selected.getValue() == expected;
            }
        });
        return std::pair{reductions, atomics};
    };

    for (AtomicAddImplementation implementation :
         {AtomicAddImplementation::Native, AtomicAddImplementation::IntegerCompareExchange}) {
        EXPECT_EQ(run(32, implementation), (std::pair<unsigned, unsigned>{0, 1}));
        EXPECT_EQ(run(64, implementation), (std::pair<unsigned, unsigned>{1, 0}));
    }
}

TEST_F(VernonStructuredVjpTest, AccumulationPolicyRejectsDeterministicAndUnsupportedSharedSums) {
    const auto rejected = [&](bool deterministic, AtomicAddImplementation implementation) {
        OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
            (Twine(R"mlir(module {
  func.func @accumulate(
      %value: f64,
      %storage: !vernon.tensor_view<f64, [1], "read_write", "device">)
      attributes {vernon.entry, vernon.stage = "compute", vernon.workgroup_size = array<i32: 64, 1, 1>} {
    %zero = arith.constant 0 : index
    "vernon.reduce_sum"(%value, %storage, %zero) {deterministic = )mlir") +
             (deterministic ? "true" : "false") +
             R"mlir(} : (f64, !vernon.tensor_view<f64, [1], "read_write", "device">, index) -> ()
    return
  }
})mlir")
                .str(),
            ParserConfig(&context));
        EXPECT_TRUE(module);
        if (!module)
            return false;
        AccumulationTargetCapabilities capabilities;
        capabilities.device.f64 = implementation;
        capabilities.supportsWorkgroupReduction = true;
        PassManager manager(&context);
        manager.addPass(createVernonLowerAccumulationPass(capabilities));
        return failed(manager.run(*module));
    };

    EXPECT_TRUE(rejected(true, AtomicAddImplementation::Native));
    EXPECT_TRUE(rejected(false, AtomicAddImplementation::Unsupported));
}

} // namespace
} // namespace mlir::vernon
