#include "runtime_rhi_test_utils.h"
#include "vernon-c/Runtime.h"

#include <cstring>
#include <gtest/gtest.h>

TEST(RuntimeCuda, CopiesAndInvokesDirectComputePipeline) {
    VernonRuntimeCapabilities capabilities = vernonRuntimeGetCapabilities(VERNON_RUNTIME_CUDA);
    if (!capabilities.available)
        GTEST_SKIP() << "CUDA runtime backend is unavailable";
    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_CUDA);
    VernonRuntimeContext *runtime = context.runtime;
    ASSERT_TRUE(runtime);
    float input[4] = {1, 2, 3, 4};
    float output[4] = {};
    auto buffer = vernon::tests::createBuffer(context, sizeof(input), alignof(float), VERNON_RHI_BUFFER_STORAGE, input);
    ASSERT_NE(buffer.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(context.device, buffer.handle, 0, output, sizeof(output)),
              VERNON_RHI_STATUS_OK);
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
    static constexpr char reflection[] = "{" VERNON_JSON_VERSION_FIELDS R"(,
    "entries": [{
      "name": "scale",
      "workgroup_size": [4, 1, 1],
      "physical_layouts": {
        "cuda_kernel_parameter": {"profile":"cuda_kernel_parameter","packing":"kernel_parameters"}
      },
      "arguments": [
        {
          "kind": "tensor",
          "dtype": "f32",
          "shape": [4],
          "element_layout": {"logical_type":"f32","byte_size":4,"alignment":4,
            "layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
            "leaves":[{"path":[],"dtype":"f32","byte_offset":0,"scalar_count":1}]},
          "physical_layouts":{"cuda_kernel_parameter":{"profile":"cuda_kernel_parameter",
            "kind":"strided_memref_storage_leaves"}}
        },
        {
          "kind": "scalar",
          "dtype": "f32",
          "element_layout": {"logical_type":"f32","byte_size":4,"alignment":4,
            "layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
            "leaves":[{"path":[],"dtype":"f32","byte_offset":0,"scalar_count":1}]},
          "physical_layouts":{"cuda_kernel_parameter":{"profile":"cuda_kernel_parameter",
            "size":4,"alignment":4,"byte_strides":[]}}
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
    arguments[0].tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    arguments[0].tensor.resource = buffer.reference;
    arguments[0].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
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
    arguments[1].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    arguments[1].tensor.access = VERNON_ACCESS_READ;
    arguments[1].tensor.byte_size = sizeof(factor);
    VernonPipelineInvocation invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PIPELINE_VERSION;
    invocation.arguments = arguments;
    invocation.argument_count = 2;
    invocation.compute_grid = {4, 1, 1};
    ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) == VERNON_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeSynchronize(runtime) == VERNON_STATUS_OK);
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(context.device, buffer.handle, 0, output, sizeof(output)),
              VERNON_RHI_STATUS_OK);
    for (int index = 0; index < 4; ++index)
        ASSERT_TRUE(output[index] == static_cast<float>(index) * factor);
    vernonRuntimeLoadedPipelineDestroy(pipeline);

    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(context.device, buffer.handle), VERNON_RHI_STATUS_OK);
    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}
