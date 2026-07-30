#include "VernonCompiler.h"
#include "VernonVersions.h"

#include <gtest/gtest.h>

#include <string>
#include <string_view>

namespace {

constexpr std::string_view synchronizationModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @synchronize(
      %output: !vernon.tensor_view<i32, [1], "write", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %storage = "vernon.workgroup_alloc"() : () ->
      !vernon.tensor_view<i32, [8], "read_write", "workgroup">
    %index = arith.constant 0 : index
    %one = arith.constant 1 : i32
    "vernon.store"(%one, %storage, %index) :
      (i32, !vernon.tensor_view<i32, [8], "read_write", "workgroup">, index) -> ()
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    %previous = "vernon.atomic"(%storage, %index, %one) {
      atomic_kind = "add", ordering = "relaxed"
    } : (!vernon.tensor_view<i32, [8], "read_write", "workgroup">, index, i32) -> i32
    "vernon.store"(%previous, %output, %index) :
      (i32, !vernon.tensor_view<i32, [1], "write", "device">, index) -> ()
    return
  }
}
)mlir";

constexpr std::string_view aggregateWorkgroupModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  "vernon.struct"() {
    sym_name = "Pair",
    fields = ["left:i32", "right:f32"],
    abi_leaf_dtypes = ["i32", "f32"]
  } : () -> ()
  func.func @aggregate_workgroup() attributes {
      vernon.entry,
      vernon.stage = "compute",
      vernon.workgroup_size = array<i32: 1, 1, 1>
    } {
    %storage = "vernon.workgroup_alloc"() : () ->
      !vernon.tensor_view<!vernon.struct<"Pair">, [2, 3], "read_write", "workgroup">
    %row = arith.constant 1 : index
    %column = arith.constant 2 : index
    %left = arith.constant 7 : i32
    %right = arith.constant 2.5 : f32
    %pair = "vernon.struct_create"(%left, %right) {type_name = "Pair"} :
      (i32, f32) -> !vernon.struct<"Pair">
    "vernon.store"(%pair, %storage, %row, %column) :
      (!vernon.struct<"Pair">,
       !vernon.tensor_view<!vernon.struct<"Pair">, [2, 3], "read_write", "workgroup">,
       index, index) -> ()
    %loaded = "vernon.load"(%storage, %row, %column) :
      (!vernon.tensor_view<!vernon.struct<"Pair">, [2, 3], "read_write", "workgroup">,
       index, index) -> !vernon.struct<"Pair">
    return
  }
}
)mlir";

constexpr std::string_view storageAtomicModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @storage_atomic(
      %values: !vernon.tensor_view<i32, [64], "read_write", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      })
      attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 64, 1, 1>
      } {
    %index = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %previous = "vernon.atomic"(%values, %index, %one) {
      atomic_kind = "add", ordering = "relaxed"
    } : (!vernon.tensor_view<i32, [64], "read_write", "device">, index, i32) -> i32
    return
  }
}
)mlir";

constexpr std::string_view invalidMemrefScopeModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @invalid_scope(
      %storage: !vernon.tensor_view<i32, [8], "read_write", "private">) {
    %index = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %previous = "vernon.atomic"(%storage, %index, %one) {
      atomic_kind = "add", ordering = "relaxed"
    } : (!vernon.tensor_view<i32, [8], "read_write", "private">, index, i32) -> i32
    return
  }
}
)mlir";

constexpr std::string_view workgroupBuiltinModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @builtin_probe(
      %local_id: tensor<3xi32> {
        vernon.interface = "input",
        vernon.builtin = "local_invocation_id"
      },
      %workgroup_id: tensor<3xi32> {
        vernon.interface = "input",
        vernon.builtin = "workgroup_id"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %index = arith.constant 0 : index
    %local_x = tensor.extract %local_id[%index] : tensor<3xi32>
    %group_x = tensor.extract %workgroup_id[%index] : tensor<3xi32>
    %sum = arith.addi %local_x, %group_x : i32
    return
  }
}
)mlir";

constexpr std::string_view nestedAggregateWorkgroupModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  "vernon.struct"() {
    sym_name = "Pair",
    fields = ["left:i32", "right:f32"],
    abi_leaf_dtypes = ["i32", "f32"]
  } : () -> ()
  func.func @nested_aggregate_workgroup(
      %lane: index {
        vernon.interface = "input",
        vernon.builtin = "local_invocation_id"
      }) attributes {
      vernon.entry,
      vernon.stage = "compute",
      vernon.workgroup_size = array<i32: 1, 1, 1>
    } {
    %storage = "vernon.workgroup_alloc"() : () ->
      !vernon.tensor_view<!vernon.struct<"Pair">, [2, 3], "read_write", "workgroup">
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %left = arith.constant 7 : i32
    %right = arith.constant 2.5 : f32
    %pair = "vernon.struct_create"(%left, %right) {type_name = "Pair"} :
      (i32, f32) -> !vernon.struct<"Pair">
    %condition = arith.cmpi eq, %lane, %zero : index
    scf.if %condition {
      "vernon.store"(%pair, %storage, %zero, %one) :
        (!vernon.struct<"Pair">,
         !vernon.tensor_view<!vernon.struct<"Pair">, [2, 3], "read_write", "workgroup">,
         index, index) -> ()
      %loaded = "vernon.load"(%storage, %zero, %one) :
        (!vernon.tensor_view<!vernon.struct<"Pair">, [2, 3], "read_write", "workgroup">,
         index, index) -> !vernon.struct<"Pair">
    }
    scf.for %iteration = %zero to %one step %one {
      "vernon.store"(%pair, %storage, %iteration, %zero) :
        (!vernon.struct<"Pair">,
         !vernon.tensor_view<!vernon.struct<"Pair">, [2, 3], "read_write", "workgroup">,
         index, index) -> ()
      %loop_loaded = "vernon.load"(%storage, %iteration, %zero) :
        (!vernon.tensor_view<!vernon.struct<"Pair">, [2, 3], "read_write", "workgroup">,
         index, index) -> !vernon.struct<"Pair">
    }
    return
  }
}
)mlir";

constexpr std::string_view combinedAggregateWorkgroupModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  "vernon.struct"() {
    sym_name = "Pair",
    fields = ["left:i32", "right:f32"],
    abi_leaf_dtypes = ["i32", "f32"]
  } : () -> ()
  func.func @combined_aggregate_workgroup() attributes {
      vernon.entry,
      vernon.stage = "compute",
      vernon.workgroup_size = array<i32: 1, 1, 1>
    } {
    %first = "vernon.workgroup_alloc"() : () ->
      !vernon.tensor_view<!vernon.struct<"Pair">, [1025], "read_write", "workgroup">
    %second = "vernon.workgroup_alloc"() : () ->
      !vernon.tensor_view<!vernon.struct<"Pair">, [1025], "read_write", "workgroup">
    return
  }
}
)mlir";

constexpr std::string_view noResultConditionalModule = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @conditional(
      %lane: index {
        vernon.interface = "input",
        vernon.builtin = "local_invocation_id"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %zero = arith.constant 0 : index
    %condition = arith.cmpi eq, %lane, %zero : index
    scf.if %condition {
    }
    return
  }
}
)mlir";

TEST(CompilerSynchronization, LowersPortableWorkgroupOperations) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    for (std::string_view atomicKind : {"add", "min", "max", "umin", "umax", "exchange"}) {
        std::string module(synchronizationModule);
        const size_t position = module.find("atomic_kind = \"add\"");
        ASSERT_NE(position, std::string::npos);
        module.replace(position, sizeof("atomic_kind = \"add\"") - 1,
                       "atomic_kind = \"" + std::string(atomicKind) + "\"");
        for (VernonTarget target : {VERNON_TARGET_CPU, VERNON_TARGET_CUDA, VERNON_TARGET_VULKAN, VERNON_TARGET_OPENGL,
                                    VERNON_TARGET_OPENGL_ES, VERNON_TARGET_METAL, VERNON_TARGET_DIRECTX}) {
            VernonCompileResult *result = vernonCompilerCompileMlir(compiler, module.data(), module.size(), target);
            ASSERT_TRUE(result);
            const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
            EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK)
                << atomicKind << ": " << std::string_view(diagnostics.data ? diagnostics.data : "", diagnostics.size);
            const VernonStringView reflection = vernonCompileResultGetReflection(result);
            const std::string_view reflected(reflection.data, reflection.size);
            EXPECT_NE(reflected.find("\"atomics\""), std::string_view::npos);
            EXPECT_NE(reflected.find("\"barriers\""), std::string_view::npos);
            EXPECT_NE(reflected.find("\"workgroup_storage\""), std::string_view::npos);
            vernonCompileResultDestroy(result);
        }
    }
    vernonCompilerDestroy(compiler);
}

TEST(CompilerSynchronization, LowersAggregateRankTwoWorkgroupStorageThroughValueAbiLeaves) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    for (VernonTarget target : {VERNON_TARGET_CPU, VERNON_TARGET_CUDA, VERNON_TARGET_VULKAN, VERNON_TARGET_OPENGL,
                                VERNON_TARGET_OPENGL_ES, VERNON_TARGET_METAL, VERNON_TARGET_DIRECTX}) {
        VernonCompileResult *result = vernonCompilerCompileMlir(compiler, aggregateWorkgroupModule.data(),
                                                                aggregateWorkgroupModule.size(), target);
        ASSERT_TRUE(result);
        const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
        EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK)
            << std::string_view(diagnostics.data ? diagnostics.data : "", diagnostics.size);
        vernonCompileResultDestroy(result);
    }
    vernonCompilerDestroy(compiler);
}

TEST(CompilerSynchronization, RejectsAggregateWorkgroupStorageAboveCanonicalLimit) {
    std::string module(aggregateWorkgroupModule);
    size_t shape = 0;
    while ((shape = module.find("[2, 3]", shape)) != std::string::npos)
        module.replace(shape, sizeof("[2, 3]") - 1, "[2049]");
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *result = vernonCompilerValidateMlir(compiler, module.data(), module.size());
    ASSERT_TRUE(result);
    EXPECT_NE(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
    const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
    EXPECT_NE(std::string_view(diagnostics.data, diagnostics.size).find("16 KiB"), std::string_view::npos);
    vernonCompileResultDestroy(result);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerSynchronization, RejectsOverflowingWorkgroupPhysicalStorage) {
    constexpr std::string_view module = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @overflow() attributes {
      vernon.entry,
      vernon.stage = "compute",
      vernon.workgroup_size = array<i32: 1, 1, 1>
    } {
    %storage = "vernon.workgroup_alloc"() : () ->
      !vernon.tensor_view<i32, [4194304, 4194304, 262144], "read_write", "workgroup">
    return
  }
}
)mlir";
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *result = vernonCompilerValidateMlir(compiler, module.data(), module.size());
    ASSERT_TRUE(result);
    const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
    EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_VERIFICATION_ERROR)
        << std::string_view(diagnostics.data ? diagnostics.data : "", diagnostics.size);
    EXPECT_NE(std::string_view(diagnostics.data, diagnostics.size).find("finite canonical physical storage plan"),
              std::string_view::npos);
    vernonCompileResultDestroy(result);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerSynchronization, RejectsCombinedWorkgroupStorageAbovePhysicalLimit) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *result = vernonCompilerValidateMlir(compiler, combinedAggregateWorkgroupModule.data(),
                                                             combinedAggregateWorkgroupModule.size());
    ASSERT_TRUE(result);
    EXPECT_NE(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
    const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
    EXPECT_NE(std::string_view(diagnostics.data, diagnostics.size).find("combined workgroup storage"),
              std::string_view::npos);
    vernonCompileResultDestroy(result);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerSynchronization, LowersNestedAggregateWorkgroupStorageUnderStructuredControlFlow) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    for (VernonTarget target : {VERNON_TARGET_CPU, VERNON_TARGET_CUDA, VERNON_TARGET_VULKAN, VERNON_TARGET_OPENGL,
                                VERNON_TARGET_OPENGL_ES, VERNON_TARGET_METAL, VERNON_TARGET_DIRECTX}) {
        VernonCompileResult *result = vernonCompilerCompileMlir(compiler, nestedAggregateWorkgroupModule.data(),
                                                                nestedAggregateWorkgroupModule.size(), target);
        ASSERT_TRUE(result);
        const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
        EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK)
            << std::string_view(diagnostics.data ? diagnostics.data : "", diagnostics.size);
        vernonCompileResultDestroy(result);
    }
    vernonCompilerDestroy(compiler);
}

TEST(CompilerSynchronization, LowersStorageTensorViewAtomicsOnlyForVerifiedTargets) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    for (std::string_view atomicKind : {"add", "min", "max", "umin", "umax", "exchange"}) {
        std::string module(storageAtomicModule);
        const size_t position = module.find("atomic_kind = \"add\"");
        ASSERT_NE(position, std::string::npos);
        module.replace(position, sizeof("atomic_kind = \"add\"") - 1,
                       "atomic_kind = \"" + std::string(atomicKind) + "\"");
        for (VernonTarget target : {VERNON_TARGET_CPU, VERNON_TARGET_CUDA, VERNON_TARGET_VULKAN}) {
            VernonCompileResult *result = vernonCompilerCompileMlir(compiler, module.data(), module.size(), target);
            ASSERT_TRUE(result);
            const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
            EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK)
                << atomicKind << ": " << std::string_view(diagnostics.data ? diagnostics.data : "", diagnostics.size);
            EXPECT_GT(vernonCompileResultGetArtifactCount(result), 0u);
            if (target == VERNON_TARGET_CUDA) {
                const VernonStringView artifact = vernonCompileResultGetArtifactData(result, 0);
                EXPECT_NE(std::string_view(artifact.data, artifact.size).find("atom"), std::string_view::npos);
            }
            vernonCompileResultDestroy(result);
        }
    }
    for (VernonTarget target :
         {VERNON_TARGET_OPENGL, VERNON_TARGET_OPENGL_ES, VERNON_TARGET_METAL, VERNON_TARGET_DIRECTX}) {
        VernonCompileResult *result =
            vernonCompilerCompileMlir(compiler, storageAtomicModule.data(), storageAtomicModule.size(), target);
        ASSERT_TRUE(result);
        EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_UNSUPPORTED_TARGET);
        EXPECT_EQ(vernonCompileResultGetArtifactCount(result), 0u);
        const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
        EXPECT_NE(std::string_view(diagnostics.data, diagnostics.size).find("supported only by CPU, CUDA, and Vulkan"),
                  std::string_view::npos);
        vernonCompileResultDestroy(result);
    }
    vernonCompilerDestroy(compiler);
}

TEST(CompilerSynchronization, RejectsCudaDeviceScopeBarrierBeforeArtifacts) {
    std::string module(synchronizationModule);
    const size_t scope = module.find("scope = \"workgroup\"");
    ASSERT_NE(scope, std::string::npos);
    module.replace(scope, sizeof("scope = \"workgroup\"") - 1, "scope = \"device\"");
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *result = vernonCompilerCompileMlir(compiler, module.data(), module.size(), VERNON_TARGET_CUDA);
    ASSERT_TRUE(result);
    EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_UNSUPPORTED_TARGET);
    EXPECT_EQ(vernonCompileResultGetArtifactCount(result), 0u);
    const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
    EXPECT_NE(std::string_view(diagnostics.data, diagnostics.size).find("device-scope barriers"),
              std::string_view::npos);
    vernonCompileResultDestroy(result);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerSynchronization, RejectsAtomicOrderingsNotImplementedByLowering) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    for (std::string_view ordering : {"acquire", "release", "acquire_release", "sequential"}) {
        std::string module(storageAtomicModule);
        const size_t position = module.find("ordering = \"relaxed\"");
        ASSERT_NE(position, std::string::npos);
        module.replace(position, sizeof("ordering = \"relaxed\"") - 1, "ordering = \"" + std::string(ordering) + "\"");

        VernonCompileResult *result = vernonCompilerValidateMlir(compiler, module.data(), module.size());
        ASSERT_TRUE(result);
        EXPECT_NE(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
        EXPECT_EQ(vernonCompileResultGetArtifactCount(result), 0u);
        const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
        EXPECT_NE(std::string_view(diagnostics.data, diagnostics.size).find("supports only relaxed memory ordering"),
                  std::string_view::npos);
        vernonCompileResultDestroy(result);
    }
    vernonCompilerDestroy(compiler);
}

TEST(CompilerSynchronization, RejectsPrivateTensorViewAtomics) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *result =
        vernonCompilerValidateMlir(compiler, invalidMemrefScopeModule.data(), invalidMemrefScopeModule.size());
    ASSERT_TRUE(result);
    EXPECT_NE(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
    const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
    EXPECT_NE(std::string_view(diagnostics.data, diagnostics.size).find("requires device or workgroup"),
              std::string_view::npos);
    vernonCompileResultDestroy(result);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerSynchronization, ValidatesNoResultConditionalWithoutElse) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *result =
        vernonCompilerValidateMlir(compiler, noResultConditionalModule.data(), noResultConditionalModule.size());
    ASSERT_TRUE(result);
    const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
    EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK)
        << std::string_view(diagnostics.data ? diagnostics.data : "", diagnostics.size);
    vernonCompileResultDestroy(result);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerSynchronization, LowersLocalAndWorkgroupBuiltins) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    for (VernonTarget target : {VERNON_TARGET_CUDA, VERNON_TARGET_VULKAN, VERNON_TARGET_OPENGL, VERNON_TARGET_OPENGL_ES,
                                VERNON_TARGET_METAL, VERNON_TARGET_DIRECTX}) {
        VernonCompileResult *result =
            vernonCompilerCompileMlir(compiler, workgroupBuiltinModule.data(), workgroupBuiltinModule.size(), target);
        ASSERT_TRUE(result);
        EXPECT_EQ(vernonCompileResultGetStatus(result), VERNON_STATUS_OK) << std::string_view(
            vernonCompileResultGetDiagnostics(result).data ? vernonCompileResultGetDiagnostics(result).data : "",
            vernonCompileResultGetDiagnostics(result).size);
        vernonCompileResultDestroy(result);
    }
    vernonCompilerDestroy(compiler);
}

} // namespace
