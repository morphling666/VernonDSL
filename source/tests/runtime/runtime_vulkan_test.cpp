#include "VernonCompiler.h"
#include "VernonRuntime.h"
#include "runtime_rhi_test_utils.h"

#include <cstring>
#include <gtest/gtest.h>

TEST(RuntimeVulkan, CompilesAndInvokesDirectComputePipeline) {
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetCapabilities(VERNON_RUNTIME_VULKAN);
    if (!capabilities.available)
        GTEST_SKIP() << "Vulkan runtime backend is unavailable";

    static constexpr char module[] = R"(
module attributes {vernon.frontend_version = 4 : i64, vernon.value_abi_version = 1 : i64} {
  func.func @increment(
      %values: !vernon.tensor_view<f32, [-1], "read_write", "device"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64,
        vernon.tensor_strides = array<i64: 1>,
        vernon.tensor_offset = 0 : i64
      },
      %id: index {
        vernon.interface = "input",
        vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %value = "vernon.load"(%values, %id) :
      (!vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> f32
    %one = arith.constant 1.0 : f32
    %sum = arith.addf %value, %one : f32
    "vernon.store"(%sum, %values, %id) :
      (f32, !vernon.tensor_view<f32, [-1], "read_write", "device">, index) -> ()
    return
  }
}
)";

    VernonCompilerContext *compiler = vernonCompilerCreate();
    ASSERT_TRUE(compiler);
    VernonCompileResult *compiled =
        vernonCompilerCompileMlir(compiler, module, std::strlen(module), VERNON_TARGET_VULKAN);
    ASSERT_TRUE(compiled);
    ASSERT_TRUE(vernonCompileResultGetStatus(compiled) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonCompileResultGetArtifactCount(compiled) == 1);

    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_VULKAN);
    VernonRuntimeContext *runtime = context.runtime;
    ASSERT_TRUE(runtime);
    VernonStringView artifact = vernonCompileResultGetArtifactData(compiled, 0);
    VernonStringView reflection = vernonCompileResultGetReflection(compiled);
    VernonLoadedPipeline *pipeline = vernonRuntimeLoadArtifact(runtime, artifact.data, artifact.size, reflection.data,
                                                               reflection.size, "increment", std::strlen("increment"));
    ASSERT_TRUE(pipeline);

    float input[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float output[8] = {};
    auto buffer = vernon::tests::createBuffer(context, sizeof(input), alignof(float), VERNON_RHI_BUFFER_STORAGE, input);
    ASSERT_NE(buffer.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    const uint64_t shape[]{8};
    const int64_t strides[]{sizeof(float)};
    VernonPipelineArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    argument.tensor.resource = buffer.reference;
    argument.tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    argument.tensor.access = VERNON_ACCESS_READ_WRITE;
    argument.tensor.rank = 1;
    argument.tensor.shape = shape;
    argument.tensor.byte_strides = strides;
    argument.tensor.byte_size = sizeof(input);
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
    invocation.arguments = &argument;
    invocation.argument_count = 1;
    invocation.compute_grid = {8, 1, 1};
    ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) == VERNON_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(context.device, buffer.handle, 0, output, sizeof(output)),
              VERNON_RHI_STATUS_OK);
    for (int index = 0; index < 8; ++index)
        ASSERT_TRUE(output[index] == input[index] + 1.0f);

    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(context.device, buffer.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
    vernonCompileResultDestroy(compiled);
    vernonCompilerDestroy(compiler);
}
