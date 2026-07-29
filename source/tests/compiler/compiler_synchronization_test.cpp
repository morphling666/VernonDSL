#include "VernonCompiler.h"

#include <gtest/gtest.h>

#include <string>
#include <string_view>

namespace {

constexpr std::string_view synchronizationModule = R"mlir(
module {
  func.func @synchronize(
      %output: !vernon.tensor_view<i32, 1, "write"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64,
        vernon.tensor_shape = array<i64: 1>,
        vernon.tensor_strides = array<i64: 1>,
        vernon.tensor_offset = 0 : i64
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 1, 1, 1>
      } {
    %storage = "vernon.workgroup_alloc"() : () -> !vernon.workgroup<i32, 8>
    %index = arith.constant 0 : index
    %one = arith.constant 1 : i32
    "vernon.workgroup_store"(%one, %storage, %index) :
      (i32, !vernon.workgroup<i32, 8>, index) -> ()
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    %previous = "vernon.atomic"(%storage, %index, %one) {
      atomic_kind = "add", ordering = "relaxed", scope = "workgroup"
    } : (!vernon.workgroup<i32, 8>, index, i32) -> i32
    "vernon.intrinsic"(%output, %index, %previous) {name = "tensor_view_store"} :
      (!vernon.tensor_view<i32, 1, "write">, index, i32) -> ()
    return
  }
}
)mlir";

constexpr std::string_view storageAtomicModule = R"mlir(
module {
  func.func @storage_atomic(
      %values: !vernon.tensor_view<i32, 1, "read_write"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64,
        vernon.tensor_shape = array<i64: 64>,
        vernon.tensor_strides = array<i64: 1>,
        vernon.tensor_offset = 0 : i64
      })
      attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 64, 1, 1>
      } {
    %index = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %previous = "vernon.atomic"(%values, %index, %one) {
      atomic_kind = "add", ordering = "relaxed", scope = "device"
    } : (!vernon.tensor_view<i32, 1, "read_write">, index, i32) -> i32
    return
  }
}
)mlir";

constexpr std::string_view invalidMemrefScopeModule = R"mlir(
module {
  func.func @invalid_scope(
      %storage: memref<8xi32, #gpu.address_space<workgroup>>) {
    %index = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %previous = "vernon.atomic"(%storage, %index, %one) {
      atomic_kind = "add", ordering = "relaxed", scope = "device"
    } : (memref<8xi32, #gpu.address_space<workgroup>>, index, i32) -> i32
    return
  }
}
)mlir";

constexpr std::string_view workgroupBuiltinModule = R"mlir(
module {
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

constexpr std::string_view noResultConditionalModule = R"mlir(
module {
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

TEST(CompilerSynchronization, InfersMemrefAtomicScopeFromMemorySpace) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *result =
        vernonCompilerValidateMlir(compiler, invalidMemrefScopeModule.data(), invalidMemrefScopeModule.size());
    ASSERT_TRUE(result);
    EXPECT_NE(vernonCompileResultGetStatus(result), VERNON_STATUS_OK);
    const VernonStringView diagnostics = vernonCompileResultGetDiagnostics(result);
    EXPECT_NE(std::string_view(diagnostics.data, diagnostics.size).find("scope must be workgroup"),
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
