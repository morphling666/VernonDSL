#include "VernonCompiler.h"
#include "VernonRuntime.h"
#include "VernonVersions.h"
#include "runtime_rhi_test_utils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <string>

namespace {

struct RhiBackendCase {
    VernonRuntimeBackend runtime;
    VernonTarget compiler;
    const char *name;
};

class RhiRuntimeSynchronization : public testing::TestWithParam<RhiBackendCase> {};

std::string stringValue(VernonStringView value) {
    return value.data ? std::string(value.data, value.size) : std::string{};
}

static constexpr char synchronizationModule[] = R"mlir(
module attributes {)mlir" VERNON_MLIR_VERSION_ATTRIBUTES R"mlir(} {
  func.func @synchronize(
      %output: !vernon.tensor_view<i32, [10], "write", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64,
        vernon.dtype = "i32",
        vernon.element_abi_leaf_dtypes = ["i32"]
      },
      %lane: index {
        vernon.interface = "input",
        vernon.builtin = "local_invocation_id"
      },
      %group: index {
        vernon.interface = "input",
        vernon.builtin = "workgroup_id"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 4, 1, 1>
      } {
    %shared = "vernon.workgroup_alloc"() : () ->
      !vernon.tensor_view<i32, [1], "read_write", "workgroup">
    %zero_index = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %hundred = arith.constant 100 : i32
    %lane_is_zero = arith.cmpi eq, %lane, %zero_index : index
    scf.if %lane_is_zero {
      %group_i32 = arith.index_cast %group : index to i32
      %group_base = arith.muli %group_i32, %hundred : i32
      "vernon.store"(%group_base, %shared, %zero_index) :
        (i32, !vernon.tensor_view<i32, [1], "read_write", "workgroup">, index) -> ()
    }
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    %previous = "vernon.atomic"(%shared, %zero_index, %one) {
      atomic_kind = "add", ordering = "relaxed"
    } : (!vernon.tensor_view<i32, [1], "read_write", "workgroup">, index, i32) -> i32
    %four = arith.constant 4 : index
    %group_offset = arith.muli %group, %four : index
    %previous_index = arith.addi %group_offset, %lane : index
    "vernon.store"(%previous, %output, %previous_index) :
      (i32, !vernon.tensor_view<i32, [10], "write", "device">, index) -> ()
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    scf.if %lane_is_zero {
      %final = "vernon.load"(%shared, %zero_index) :
        (!vernon.tensor_view<i32, [1], "read_write", "workgroup">, index) -> i32
      %eight = arith.constant 8 : index
      %final_index = arith.addi %eight, %group : index
      "vernon.store"(%final, %output, %final_index) :
        (i32, !vernon.tensor_view<i32, [10], "write", "device">, index) -> ()
    }
    return
  }
}
)mlir";

void verifySynchronizationResult(const std::array<int32_t, 10> &result) {
    for (size_t group = 0; group < 2; ++group) {
        std::array<int32_t, 4> previous{result[group * 4], result[group * 4 + 1], result[group * 4 + 2],
                                        result[group * 4 + 3]};
        std::sort(previous.begin(), previous.end());
        const int32_t base = static_cast<int32_t>(group * 100);
        EXPECT_EQ(previous, (std::array<int32_t, 4>{base, base + 1, base + 2, base + 3}));
        EXPECT_EQ(result[8 + group], base + 4);
    }
}

VernonCompileResult *compileSynchronization(VernonTarget target, VernonCompilerContext **compiler) {
    *compiler = vernonCompilerCreate();
    if (!*compiler)
        return nullptr;
    if (target == VERNON_TARGET_CPU) {
        const VernonCpuRuntimeHelpersV1 helpers{sizeof(VernonCpuRuntimeHelpersV1), &vernonCpuWorkgroupAddressV1,
                                                &vernonCpuLaneAddressV1, &vernonCpuWorkgroupBarrierV1,
                                                &vernonCpuWorkgroupIsLeaderV1};
        if (vernonCompilerRegisterCpuRuntimeHelpersV1(*compiler, &helpers) != VERNON_STATUS_OK)
            return nullptr;
    }
    return vernonCompilerCompileMlir(*compiler, synchronizationModule, sizeof(synchronizationModule) - 1, target);
}

TEST(CompilerRuntimeSynchronization, CpuJitEntryExecutesCooperativeWorkgroups) {
    VernonCompilerContext *compiler = nullptr;
    VernonCompileResult *compiled = compileSynchronization(VERNON_TARGET_CPU, &compiler);
    ASSERT_NE(compiler, nullptr);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(vernonCompileResultGetStatus(compiled), VERNON_STATUS_OK)
        << stringValue(vernonCompileResultGetDiagnostics(compiled));
    ASSERT_EQ(vernonCompileResultGetArtifactCount(compiled), 1u);

    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(runtime, nullptr);
    const VernonStringView reflection = vernonCompileResultGetReflection(compiled);
    VernonCpuEntryPoint entry = vernonCompileResultGetCpuEntry(compiled, "synchronize", std::strlen("synchronize"));
    ASSERT_NE(entry, nullptr);
    VernonStageExecutable *pipeline = vernonRuntimeLoadCpuEntry(runtime, entry, reflection.data, reflection.size,
                                                                "synchronize", std::strlen("synchronize"));
    ASSERT_NE(pipeline, nullptr) << stringValue(vernonRuntimeGetLastError(runtime));

    std::array<int32_t, 10> result{};
    constexpr uint64_t shape[]{result.size()};
    constexpr int64_t strides[]{sizeof(int32_t)};
    VernonProgramArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_HOST;
    argument.tensor.host_data = result.data();
    argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_I32);
    argument.tensor.access = VERNON_ACCESS_WRITE;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = sizeof(result);
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.compute_grid = {2, 1, 1};
    ASSERT_EQ(vernon::tests::completeSubmission(pipeline, &invocation), VERNON_STATUS_OK)
        << stringValue(vernonRuntimeGetLastError(runtime));
    verifySynchronizationResult(result);

    vernonRuntimeStageExecutableDestroy(pipeline);
    EXPECT_EQ(vernonRuntimeDestroy(runtime), VERNON_STATUS_OK);
    vernonCompileResultDestroy(compiled);
    vernonCompilerDestroy(compiler);
}

TEST(CompilerRuntimeSynchronization, CpuArtifactCompilationDoesNotUseProcessGlobalHelpers) {
    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_NE(compiler, nullptr);
    VernonCompileResult *compiled = vernonCompilerCompileMlir(compiler, synchronizationModule,
                                                              sizeof(synchronizationModule) - 1, VERNON_TARGET_CPU);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(vernonCompileResultGetStatus(compiled), VERNON_STATUS_OK)
        << stringValue(vernonCompileResultGetDiagnostics(compiled));
    EXPECT_EQ(vernonCompileResultGetArtifactCount(compiled), 1u);
    EXPECT_EQ(vernonCompileResultGetCpuEntry(compiled, "synchronize", std::strlen("synchronize")), nullptr);
    vernonCompileResultDestroy(compiled);
    vernonCompilerDestroy(compiler);
}

TEST_P(RhiRuntimeSynchronization, ExecutesIndependentWorkgroupBarrierAndAtomic) {
    const RhiBackendCase backend = GetParam();
    if (!vernonRuntimeGetCapabilities(backend.runtime).available)
        GTEST_SKIP() << backend.name << " runtime backend is unavailable";

    VernonCompilerContext *compiler = nullptr;
    VernonCompileResult *compiled = compileSynchronization(backend.compiler, &compiler);
    ASSERT_NE(compiler, nullptr);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(vernonCompileResultGetStatus(compiled), VERNON_STATUS_OK)
        << stringValue(vernonCompileResultGetDiagnostics(compiled));
    ASSERT_EQ(vernonCompileResultGetArtifactCount(compiled), 1u);

    auto context =
        vernon::tests::createRhiRuntime(backend.runtime, nullptr, backend.runtime == VERNON_RUNTIME_DIRECTX12);
    ASSERT_NE(context.runtime, nullptr);
    const VernonStringView artifact = vernonCompileResultGetArtifactData(compiled, 0);
    const VernonStringView reflection = vernonCompileResultGetReflection(compiled);
    VernonStageExecutable *pipeline =
        vernonRuntimeLoadArtifact(context.runtime, artifact.data, artifact.size, reflection.data, reflection.size,
                                  "synchronize", std::strlen("synchronize"));
    ASSERT_NE(pipeline, nullptr) << stringValue(vernonRuntimeGetLastError(context.runtime));

    std::array<int32_t, 10> result{};
    vernon::tests::RhiBuffer buffer = vernon::tests::createBuffer(context, sizeof(result), alignof(int32_t),
                                                                  VERNON_RHI_BUFFER_STORAGE, result.data());
    ASSERT_NE(buffer.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    constexpr uint64_t shape[]{result.size()};
    constexpr int64_t strides[]{sizeof(int32_t)};
    VernonProgramArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PROGRAM_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    argument.tensor.resource = buffer.reference;
    argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_I32);
    argument.tensor.access = VERNON_ACCESS_WRITE;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = sizeof(result);
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.compute_grid = {2, 1, 1};
    ASSERT_EQ(vernon::tests::completeSubmission(pipeline, &invocation), VERNON_STATUS_OK)
        << "RHI: " << stringValue(vernonRhiDeviceGetLastError(context.device))
        << "; runtime: " << stringValue(vernonRuntimeGetLastError(context.runtime));
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(context.device, buffer.handle, 0, result.data(), sizeof(result)),
              VERNON_RHI_STATUS_OK);
    verifySynchronizationResult(result);

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(context.device, buffer.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeStageExecutableDestroy(pipeline);
    EXPECT_EQ(vernonRuntimeDestroy(context.runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
    vernonCompileResultDestroy(compiled);
    vernonCompilerDestroy(compiler);
}

INSTANTIATE_TEST_SUITE_P(AvailableBackends, RhiRuntimeSynchronization,
                         testing::Values(RhiBackendCase{VERNON_RUNTIME_CUDA, VERNON_TARGET_CUDA, "CUDA"},
                                         RhiBackendCase{VERNON_RUNTIME_VULKAN, VERNON_TARGET_VULKAN, "Vulkan"},
                                         RhiBackendCase{VERNON_RUNTIME_OPENGL, VERNON_TARGET_OPENGL, "OpenGL"},
                                         RhiBackendCase{VERNON_RUNTIME_DIRECTX12, VERNON_TARGET_DIRECTX, "DirectX"}),
                         [](const testing::TestParamInfo<RhiBackendCase> &info) { return info.param.name; });

} // namespace
