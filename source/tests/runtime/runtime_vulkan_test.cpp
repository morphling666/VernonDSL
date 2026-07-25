#include "VernonCompiler.h"
#include "VernonRuntime.h"

#include <cstring>
#include <gtest/gtest.h>

TEST(RuntimeVulkan, CompilesAndInvokesDirectComputePipeline) {
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetCapabilities(VERNON_RUNTIME_VULKAN);
    if (!capabilities.available)
        GTEST_SKIP() << "Vulkan runtime backend is unavailable";

    static constexpr char module[] = R"(
module {
  func.func @increment(
      %values: !vernon.tensor_view<f32, 1, "read_write"> {
        vernon.interface = "resource",
        vernon.set = 0 : i64,
        vernon.binding = 0 : i64
      },
      %id: index {
        vernon.interface = "input",
        vernon.builtin = "global_invocation_id"
      }) attributes {
        vernon.entry,
        vernon.stage = "compute",
        vernon.workgroup_size = array<i32: 8, 1, 1>
      } {
    %value = "vernon.intrinsic"(%values, %id) {
      name = "tensor_view_load"
    } : (!vernon.tensor_view<f32, 1, "read_write">, index) -> f32
    %one = arith.constant 1.0 : f32
    %sum = arith.addf %value, %one : f32
    "vernon.intrinsic"(%values, %id, %sum) {
      name = "tensor_view_store"
    } : (!vernon.tensor_view<f32, 1, "read_write">, index, f32) -> ()
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

    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_VULKAN, nullptr);
    ASSERT_TRUE(runtime);
    VernonStringView artifact = vernonCompileResultGetArtifactData(compiled, 0);
    VernonStringView reflection = vernonCompileResultGetReflection(compiled);
    VernonLoadedPipeline *pipeline = vernonRuntimeLoadArtifact(runtime, artifact.data, artifact.size, reflection.data,
                                                               reflection.size, "increment", std::strlen("increment"));
    ASSERT_TRUE(pipeline);

    float input[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float output[8] = {};
    VernonDeviceBuffer *buffer = vernonRuntimeBufferAllocate(runtime, sizeof(input), alignof(float));
    ASSERT_TRUE(buffer);
    ASSERT_TRUE(vernonRuntimeCopyFromHost(buffer, 0, input, sizeof(input)) == VERNON_STATUS_OK);
    const uint64_t shape[]{8};
    const int64_t strides[]{sizeof(float)};
    VernonPipelineArgument argument{};
    argument.slot = 0;
    argument.kind = VERNON_PIPELINE_TENSOR;
    argument.tensor.struct_size = sizeof(VernonTensorView);
    argument.tensor.storage = VERNON_TENSOR_DEVICE;
    argument.tensor.buffer = buffer;
    argument.tensor.dtype = VERNON_DATA_F32;
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
    ASSERT_TRUE(vernonRuntimeCopyToHost(buffer, 0, output, sizeof(output)) == VERNON_STATUS_OK);
    for (int index = 0; index < 8; ++index)
        ASSERT_TRUE(output[index] == input[index] + 1.0f);

    ASSERT_TRUE(vernonRuntimeBufferFree(buffer) == VERNON_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
    vernonCompileResultDestroy(compiled);
    vernonCompilerDestroy(compiler);
}
