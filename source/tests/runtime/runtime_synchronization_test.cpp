#include "VernonCompiler.h"
#include "VernonRuntime.h"
#include "runtime_rhi_test_utils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <string>

namespace {

struct BackendCase {
    VernonRuntimeBackend runtime;
    VernonTarget compiler;
    const char *name;
};

class RuntimeSynchronization : public testing::TestWithParam<BackendCase> {};

std::string stringValue(VernonStringView value) {
    return value.data ? std::string(value.data, value.size) : std::string{};
}

TEST_P(RuntimeSynchronization, ExecutesIndependentWorkgroupBarrierAndAtomic) {
    const BackendCase backend = GetParam();
    if (!vernonRuntimeGetCapabilities(backend.runtime).available)
        GTEST_SKIP() << backend.name << " runtime backend is unavailable";

    static constexpr char module[] = R"mlir(
module {
  func.func @synchronize(
      %output: !vernon.tensor_view<i32, 1, "write"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64,
        vernon.tensor_shape = array<i64: 10>,
        vernon.tensor_strides = array<i64: 1>,
        vernon.tensor_offset = 0 : i64
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
    %shared = "vernon.workgroup_alloc"() : () -> !vernon.workgroup<i32, 1>
    %zero_index = arith.constant 0 : index
    %one = arith.constant 1 : i32
    %hundred = arith.constant 100 : i32
    %lane_is_zero = arith.cmpi eq, %lane, %zero_index : index
    scf.if %lane_is_zero {
      %group_i32 = arith.index_cast %group : index to i32
      %group_base = arith.muli %group_i32, %hundred : i32
      "vernon.workgroup_store"(%group_base, %shared, %zero_index) :
        (i32, !vernon.workgroup<i32, 1>, index) -> ()
    }
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    %previous = "vernon.atomic"(%shared, %zero_index, %one) {
      atomic_kind = "add", ordering = "relaxed", scope = "workgroup"
    } : (!vernon.workgroup<i32, 1>, index, i32) -> i32
    %four = arith.constant 4 : index
    %group_offset = arith.muli %group, %four : index
    %previous_index = arith.addi %group_offset, %lane : index
    "vernon.intrinsic"(%output, %previous_index, %previous) {name = "tensor_view_store"} :
      (!vernon.tensor_view<i32, 1, "write">, index, i32) -> ()
    "vernon.barrier"() {ordering = "acquire_release", scope = "workgroup"} : () -> ()
    scf.if %lane_is_zero {
      %final = "vernon.workgroup_load"(%shared, %zero_index) :
        (!vernon.workgroup<i32, 1>, index) -> i32
      %eight = arith.constant 8 : index
      %final_index = arith.addi %eight, %group : index
      "vernon.intrinsic"(%output, %final_index, %final) {name = "tensor_view_store"} :
        (!vernon.tensor_view<i32, 1, "write">, index, i32) -> ()
    }
    return
  }
}
)mlir";

    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_NE(compiler, nullptr);
    VernonCompileResult *compiled = vernonCompilerCompileMlir(compiler, module, sizeof(module) - 1, backend.compiler);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(vernonCompileResultGetStatus(compiled), VERNON_STATUS_OK)
        << stringValue(vernonCompileResultGetDiagnostics(compiled));
    ASSERT_EQ(vernonCompileResultGetArtifactCount(compiled), 1u);

    auto context =
        vernon::tests::createRhiRuntime(backend.runtime, nullptr, backend.runtime == VERNON_RUNTIME_DIRECTX12);
    ASSERT_NE(context.runtime, nullptr);
    const VernonStringView artifact = vernonCompileResultGetArtifactData(compiled, 0);
    const VernonStringView reflection = vernonCompileResultGetReflection(compiled);
    VernonLoadedPipeline *pipeline =
        vernonRuntimeLoadArtifact(context.runtime, artifact.data, artifact.size, reflection.data, reflection.size,
                                  "synchronize", std::strlen("synchronize"));
    ASSERT_NE(pipeline, nullptr) << stringValue(vernonRuntimeGetLastError(context.runtime));

    constexpr std::array<int32_t, 10> initial{};
    auto buffer = vernon::tests::createBuffer(context, sizeof(initial), alignof(int32_t), VERNON_RHI_BUFFER_STORAGE,
                                              initial.data());
    ASSERT_NE(buffer.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    constexpr uint64_t shape[]{initial.size()};
    constexpr int64_t strides[]{sizeof(int32_t)};
    VernonPipelineArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    argument.tensor.resource = buffer.reference;
    argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_I32);
    argument.tensor.access = VERNON_ACCESS_WRITE;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = sizeof(initial);
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.compute_grid = {8, 1, 1};
    ASSERT_EQ(vernonRuntimePipelineInvoke(pipeline, &invocation), VERNON_STATUS_OK)
        << stringValue(vernonRuntimeGetLastError(context.runtime));
    ASSERT_EQ(vernonRuntimeSynchronize(context.runtime), VERNON_STATUS_OK);

    std::array<int32_t, 10> result{};
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(context.device, buffer.handle, 0, result.data(), sizeof(result)),
              VERNON_RHI_STATUS_OK);
    for (size_t group = 0; group < 2; ++group) {
        std::array<int32_t, 4> previous{result[group * 4], result[group * 4 + 1], result[group * 4 + 2],
                                        result[group * 4 + 3]};
        std::sort(previous.begin(), previous.end());
        const int32_t base = static_cast<int32_t>(group * 100);
        EXPECT_EQ(previous, (std::array<int32_t, 4>{base, base + 1, base + 2, base + 3}));
        EXPECT_EQ(result[8 + group], base + 4);
    }

    EXPECT_EQ(vernonRhiDeviceDestroyBuffer(context.device, buffer.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    EXPECT_EQ(vernonRuntimeDestroy(context.runtime), VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
    vernonCompileResultDestroy(compiled);
    vernonCompilerDestroy(compiler);
}

INSTANTIATE_TEST_SUITE_P(AvailableBackends, RuntimeSynchronization,
                         testing::Values(BackendCase{VERNON_RUNTIME_CUDA, VERNON_TARGET_CUDA, "CUDA"},
                                         BackendCase{VERNON_RUNTIME_VULKAN, VERNON_TARGET_VULKAN, "Vulkan"},
                                         BackendCase{VERNON_RUNTIME_OPENGL, VERNON_TARGET_OPENGL, "OpenGL"},
                                         BackendCase{VERNON_RUNTIME_DIRECTX12, VERNON_TARGET_DIRECTX, "DirectX"}),
                         [](const testing::TestParamInfo<BackendCase> &info) { return info.param.name; });

} // namespace
