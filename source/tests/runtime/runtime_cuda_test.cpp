#include "vernon-c/Runtime.h"

#include <cstring>
#include <gtest/gtest.h>

TEST(RuntimeCuda, CopiesAndInvokesDirectComputePipeline) {
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetCapabilities(VERNON_RUNTIME_CUDA);
    if (!capabilities.available)
        GTEST_SKIP() << "CUDA runtime backend is unavailable";
    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CUDA, nullptr);
    ASSERT_TRUE(runtime);
    VernonDeviceBuffer *buffer = vernonRuntimeBufferAllocate(runtime, 16, alignof(float));
    ASSERT_TRUE(buffer);
    float input[4] = {1, 2, 3, 4};
    float output[4] = {};
    ASSERT_TRUE(vernonRuntimeCopyFromHost(buffer, 0, input, sizeof(input)) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeCopyToHost(buffer, 0, output, sizeof(output)) == VERNON_STATUS_OK);
    for (int index = 0; index < 4; ++index)
        ASSERT_TRUE(input[index] == output[index]);

    static constexpr char ptx[] = R"(
.version 8.0
.target sm_80
.address_size 64
.visible .entry scale(
  .param .u64 allocated,
  .param .u64 aligned,
  .param .u64 offset,
  .param .u64 size,
  .param .u64 stride,
  .param .f32 factor
) {
  .reg .b32 %r<3>;
  .reg .b64 %rd<4>;
  .reg .f32 %f<3>;
  ld.param.u64 %rd0, [aligned];
  cvta.to.global.u64 %rd1, %rd0;
  ld.param.f32 %f0, [factor];
  mov.u32 %r0, %tid.x;
  cvt.rn.f32.u32 %f1, %r0;
  mul.f32 %f2, %f1, %f0;
  mul.wide.u32 %rd2, %r0, 4;
  add.u64 %rd3, %rd1, %rd2;
  st.global.f32 [%rd3], %f2;
  ret;
}
)";
    static constexpr char reflection[] = R"({
    "gpu_launch_abi_version": 1,
    "entries": [{
      "name": "scale",
      "workgroup_size": [4, 1, 1],
      "cpu_arguments_size": 12,
      "arguments": [
        {
          "kind": "tensor",
          "dtype": "f32",
          "shape": [4],
          "alignment": 4,
          "cpu_offset": 0,
          "cpu_size": 8
        },
        {
          "kind": "scalar",
          "dtype": "f32",
          "alignment": 4,
          "cpu_offset": 8,
          "cpu_size": 4
        }
      ]
    }]
  })";
    VernonLoadedPipeline *pipeline =
        vernonRuntimeLoadArtifact(runtime, ptx, std::strlen(ptx), reflection, std::strlen(reflection), "scale", 5);
    ASSERT_TRUE(pipeline);
    const float factor = 2.0f;
    const uint64_t shape[]{4};
    const int64_t strides[]{sizeof(float)};
    VernonPipelineArgument arguments[2]{};
    arguments[0].slot = 0;
    arguments[0].kind = VERNON_PIPELINE_TENSOR;
    arguments[0].tensor.struct_size = sizeof(VernonTensorView);
    arguments[0].tensor.storage = VERNON_TENSOR_DEVICE;
    arguments[0].tensor.buffer = buffer;
    arguments[0].tensor.dtype = VERNON_DATA_F32;
    arguments[0].tensor.access = VERNON_ACCESS_READ_WRITE;
    arguments[0].tensor.rank = 1;
    arguments[0].tensor.shape = shape;
    arguments[0].tensor.byte_strides = strides;
    arguments[0].tensor.byte_size = sizeof(output);
    arguments[1].slot = 1;
    arguments[1].kind = VERNON_PIPELINE_TENSOR;
    arguments[1].tensor.struct_size = sizeof(VernonTensorView);
    arguments[1].tensor.storage = VERNON_TENSOR_HOST;
    arguments[1].tensor.host_data = &factor;
    arguments[1].tensor.dtype = VERNON_DATA_F32;
    arguments[1].tensor.access = VERNON_ACCESS_READ;
    arguments[1].tensor.byte_size = sizeof(factor);
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
    invocation.arguments = arguments;
    invocation.argument_count = 2;
    invocation.compute_grid = {4, 1, 1};
    ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeSynchronize(runtime) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeCopyToHost(buffer, 0, output, sizeof(output)) == VERNON_STATUS_OK);
    for (int index = 0; index < 4; ++index)
        ASSERT_TRUE(output[index] == static_cast<float>(index) * factor);
    vernonRuntimeLoadedPipelineDestroy(pipeline);

    ASSERT_TRUE(vernonRuntimeBufferFree(buffer) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
}
