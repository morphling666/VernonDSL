#include "runtime_rhi_test_utils.h"
#include "vernon-c/Runtime.h"

#include <cstring>
#include <gtest/gtest.h>
#include <string>

TEST(RuntimeCudaPipeline, LoadsAndInvokesBundle) {
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
      "dispatch_contract": {"unit_grid_axes": [], "requires_unit_workgroup": false},
      "physical_layouts": {
        "cuda_kernel_parameter": {"profile":"cuda_kernel_parameter","packing":"kernel_parameters"}
      },
      "arguments": [
        {"kind": "tensor", "dtype": "f32", "shape": [4],
         "vernon.source_name":"output","vernon.source_access":"write",
         "element_layout": {"logical_type":"f32","byte_size":4,"alignment":4,
          "layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
          "leaves":[{"path":[],"dtype":"f32","byte_offset":0,"scalar_count":1}]},
         "physical_layouts":{"cuda_kernel_parameter":{"profile":"cuda_kernel_parameter",
          "kind":"strided_memref_storage_leaves"}}},
        {"kind": "scalar", "dtype": "f32",
         "vernon.source_name":"factor","vernon.source_access":"read",
         "element_layout": {"logical_type":"f32","byte_size":4,"alignment":4,
          "layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
          "leaves":[{"path":[],"dtype":"f32","byte_offset":0,"scalar_count":1}]},
         "physical_layouts":{"cuda_kernel_parameter":{"profile":"cuda_kernel_parameter",
          "size":4,"alignment":4,"byte_strides":[]}}}
      ]
    }]
  })";
    if (!vernonRuntimeGetCapabilities(VERNON_RUNTIME_CUDA).available) {
        GTEST_SKIP() << "CUDA runtime backend is unavailable";
    }

    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_CUDA);
    VernonRuntimeContext *runtime = context.runtime;
    ASSERT_TRUE(runtime);
    VernonStageExecutable *pipeline = vernonRuntimeLoadArtifact(runtime, ptx, sizeof(ptx) - 1, reflection,
                                                                sizeof(reflection) - 1, "scale", std::strlen("scale"));
    ASSERT_TRUE(pipeline) << std::string(vernonRuntimeGetLastError(runtime).data,
                                         vernonRuntimeGetLastError(runtime).size);

    ASSERT_TRUE(vernonRuntimeStageExecutableGetParameterCount(pipeline) == 2);
    VernonProgramParameterView parameter{};
    ASSERT_TRUE(vernonRuntimeStageExecutableGetParameterByIndex(pipeline, 0, &parameter) == VERNON_STATUS_OK);
    ASSERT_TRUE(parameter.slot == 0 && parameter.kind == VERNON_PROGRAM_TENSOR &&
                parameter.element_layout.leaf_count == 1 &&
                parameter.element_layout.leaves[0].dtype == VERNON_DATA_F32 &&
                parameter.access == VERNON_ACCESS_WRITE && parameter.rank == 1 && parameter.static_shape[0] == 4);
    ASSERT_TRUE(vernonRuntimeStageExecutableFindParameter(pipeline, {"factor", std::strlen("factor")}, &parameter) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(parameter.slot == 1 && parameter.kind == VERNON_PROGRAM_TENSOR);
    auto buffer = vernon::tests::createBuffer(context, 4 * sizeof(float), alignof(float), VERNON_RHI_BUFFER_STORAGE);
    ASSERT_NE(buffer.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    const uint64_t shape[] = {4};
    const int64_t strides[] = {sizeof(float)};
    const float factor = 3.0f;
    VernonProgramArgument arguments[2]{};
    arguments[0].slot = 0;
    arguments[0].kind = VERNON_PROGRAM_TENSOR;
    arguments[0].tensor.struct_size = sizeof(VernonTensorView);
    arguments[0].tensor.storage = VERNON_TENSOR_RHI_RESOURCE;
    arguments[0].tensor.resource = buffer.reference;
    arguments[0].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    arguments[0].tensor.access = VERNON_ACCESS_WRITE;
    arguments[0].tensor.rank = 1;
    arguments[0].tensor.shape = shape;
    arguments[0].tensor.byte_strides = strides;
    arguments[0].tensor.byte_size = 4 * sizeof(float);
    arguments[1].slot = 1;
    arguments[1].kind = VERNON_PROGRAM_TENSOR;
    arguments[1].tensor.struct_size = sizeof(VernonTensorView);
    arguments[1].tensor.storage = VERNON_TENSOR_HOST;
    arguments[1].tensor.host_data = &factor;
    arguments[1].tensor.element_layout = vernonRuntimeGetScalarValueLayout(VERNON_DATA_F32);
    arguments[1].tensor.access = VERNON_ACCESS_READ;
    arguments[1].tensor.byte_size = sizeof(factor);
    VernonStageInvocationDescriptor invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.abi_version = VERNON_PROGRAM_VERSION;
    invocation.arguments = arguments;
    invocation.argument_count = 2;
    invocation.compute_grid = {4, 1, 1};
    ASSERT_TRUE(vernon::tests::completeSubmission(pipeline, &invocation) == VERNON_STATUS_OK);

    float output[4]{};
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(context.device, buffer.handle, 0, output, sizeof(output)),
              VERNON_RHI_STATUS_OK);
    for (int index = 0; index < 4; ++index)
        ASSERT_TRUE(output[index] == static_cast<float>(index) * factor);

    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(context.device, buffer.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeStageExecutableDestroy(pipeline);
    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}
