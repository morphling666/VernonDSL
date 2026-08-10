#include "runtime/content_hash.h"
#include "runtime_rhi_test_utils.h"
#include "vernon-c/Runtime.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <string>

namespace {

std::string jsonString(const char *value) {
    std::string result{"\""};
    for (; *value; ++value) {
        if (*value == '\\' || *value == '"')
            result.push_back('\\');
        if (*value == '\n') {
            result += "\\n";
            continue;
        }
        result.push_back(*value);
    }
    result.push_back('"');
    return result;
}

std::string withContentHash(nlohmann::json root) {
    root.erase("content_hash");
    const std::string canonical = root.dump(-1, ' ', false);
    root["content_hash"] = vernon::runtime::sha256Hex(canonical.data(), canonical.size());
    return root.dump(-1, ' ', false);
}

} // namespace

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
         "element_layout": {"logical_type":"f32","byte_size":4,"alignment":4,
          "layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
          "leaves":[{"path":[],"dtype":"f32","byte_offset":0,"scalar_count":1}]},
         "physical_layouts":{"cuda_kernel_parameter":{"profile":"cuda_kernel_parameter",
          "kind":"strided_memref_storage_leaves"}}},
        {"kind": "scalar", "dtype": "f32",
         "element_layout": {"logical_type":"f32","byte_size":4,"alignment":4,
          "layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",
          "leaves":[{"path":[],"dtype":"f32","byte_offset":0,"scalar_count":1}]},
         "physical_layouts":{"cuda_kernel_parameter":{"profile":"cuda_kernel_parameter",
          "size":4,"alignment":4,"byte_strides":[]}}}
      ]
    }]
  })";
    const std::string ptxHash = vernon::runtime::sha256Hex(ptx, sizeof(ptx) - 1);
    std::string bundle =
        "{" VERNON_PIPELINE_JSON_FIELD R"(,)"
        R"("type":"pipeline","id":"cuda/scale","target":{"kind":"cuda","options":{}},)"
        R"("runtime_requirements":{"backend":"cuda",)"
        R"("features":[],"ptx_version":[8,0],)"
        R"("minimum_compute_capability":[8,0],"address_size":64},)"
        R"("variants":[{"key":[],"parameters":[)"
        R"({"slot":0,"name":"output","kind":"tensor",)"
        R"("type":"!vernon.tensor_view<f32, [4], \"write\", \"device\">",)"
        R"("address_space":"device","element_layout":)"
        R"({"logical_type":"f32","byte_size":4,"alignment":4,)"
        R"("layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",)"
        R"("leaves":[{"path":[],"dtype":"f32","byte_offset":0,"scalar_count":1}]},)"
        R"("shape":[4],"access":"write","uses":[{"stage":"compute",)"
        R"("index":0,"dtype":"f32","shape":[4],"interface":"storage"}]},)"
        R"({"slot":1,"name":"factor","kind":"tensor","type":"f32","value_layout":)"
        R"({"logical_type":"f32","byte_size":4,"alignment":4,)"
        R"("layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",)"
        R"("leaves":[{"path":[],"dtype":"f32","byte_offset":0,"scalar_count":1}]},)"
        R"("shape":[],"access":"read","uses":[{"stage":"compute",)"
        R"("index":1,"dtype":"f32","shape":[],"interface":"value",)"
        R"("interface_plan":{"kind":"kernel_parameter","profile":"cuda_kernel_parameter",)"
        R"("canonical_layout_hash":"cb580e347f23fbe3afbd1c5f72b4d2339b09e33d876f79e9d290445edb43c03b",)"
        R"("root":{"kind":"scalar","representation":"f32","offset":0,"size":4,)"
        R"("alignment":4,"shape":[],"byte_strides":[],"children":[]}}}]})"
        R"(],"outputs":[{"name":"result","kind":"tensor","dtype":"f32",)"
        R"("shape":[4],"access":"write","location":0}],)"
        R"("program":{"compute":"scale"}}],)"
        R"("stage_artifacts":{"scale":{"entry":"scale",)"
        R"("stage":"compute","artifact":{)"
        R"("format":"ptx","storage":"inline","encoding":"utf8","data":)" +
        jsonString(ptx) + R"(,"size":)" + std::to_string(sizeof(ptx) - 1) + R"(,"sha256":)" +
        jsonString(ptxHash.c_str()) + R"(},"reflection":)" + reflection + "}}}";
    bundle = withContentHash(nlohmann::json::parse(bundle));

    VernonRuntimeBackend target = VERNON_RUNTIME_CPU;
    ASSERT_TRUE(vernonRuntimePipelineBundleInspectTarget(bundle.data(), bundle.size(), &target) == VERNON_STATUS_OK);
    ASSERT_TRUE(target == VERNON_RUNTIME_CUDA);

    if (!vernonRuntimeGetCapabilities(VERNON_RUNTIME_CUDA).available) {
        GTEST_SKIP() << "CUDA runtime backend is unavailable";
    }

    auto context = vernon::tests::createRhiRuntime(VERNON_RUNTIME_CUDA);
    VernonRuntimeContext *runtime = context.runtime;
    ASSERT_TRUE(runtime);
    VernonPipelineBundleLoadOptions options{};
    options.struct_size = sizeof(options);

    nlohmann::json unsupportedDocument = nlohmann::json::parse(bundle);
    unsupportedDocument["runtime_requirements"]["minimum_compute_capability"] = nlohmann::json::array({99, 0});
    const std::string unsupported = withContentHash(std::move(unsupportedDocument));
    ASSERT_FALSE(vernonRuntimeLoadPipelineBundleWithOptions(runtime, unsupported.data(), unsupported.size(), &options));
    const VernonStringView unsupportedError = vernonRuntimeGetLastError(runtime);
    ASSERT_TRUE(std::string(unsupportedError.data, unsupportedError.size)
                    .find("pipeline requires CUDA compute capability 99.0, device provides") != std::string::npos);

    VernonPipelineBundle *loaded =
        vernonRuntimeLoadPipelineBundleWithOptions(runtime, bundle.data(), bundle.size(), &options);
    ASSERT_TRUE(loaded);
    VernonLoadedPipeline *pipeline = vernonRuntimeResolvePipeline(loaded, {nullptr, 0});
    ASSERT_TRUE(pipeline);

    ASSERT_TRUE(vernonRuntimeLoadedPipelineGetParameterCount(pipeline) == 2);
    VernonPipelineParameterView parameter{};
    ASSERT_TRUE(vernonRuntimeLoadedPipelineGetParameterByIndex(pipeline, 0, &parameter) == VERNON_STATUS_OK);
    ASSERT_TRUE(parameter.slot == 0 && parameter.kind == VERNON_PIPELINE_TENSOR &&
                parameter.element_layout.leaf_count == 1 &&
                parameter.element_layout.leaves[0].dtype == VERNON_DATA_F32 &&
                parameter.access == VERNON_ACCESS_WRITE && parameter.rank == 1 && parameter.static_shape[0] == 4);
    ASSERT_TRUE(vernonRuntimeLoadedPipelineFindParameter(pipeline, {"factor", std::strlen("factor")}, &parameter) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(parameter.slot == 1 && parameter.kind == VERNON_PIPELINE_TENSOR);
    ASSERT_TRUE(vernonRuntimeLoadedPipelineGetOutputCount(pipeline) == 1);
    VernonPipelineOutputView pipelineOutput{};
    ASSERT_TRUE(vernonRuntimeLoadedPipelineFindOutput(pipeline, {"result", std::strlen("result")}, &pipelineOutput) ==
                VERNON_STATUS_OK);
    ASSERT_TRUE(pipelineOutput.kind == VERNON_PIPELINE_TENSOR && pipelineOutput.dtype == VERNON_DATA_F32 &&
                pipelineOutput.access == VERNON_ACCESS_WRITE && pipelineOutput.rank == 1 &&
                pipelineOutput.static_shape[0] == 4 && pipelineOutput.location == 0);

    auto buffer = vernon::tests::createBuffer(context, 4 * sizeof(float), alignof(float), VERNON_RHI_BUFFER_STORAGE);
    ASSERT_NE(buffer.handle.index, VERNON_RHI_INVALID_HANDLE_INDEX);
    const uint64_t shape[] = {4};
    const int64_t strides[] = {sizeof(float)};
    const float factor = 3.0f;
    VernonPipelineArgument arguments[2]{};
    arguments[0].slot = 0;
    arguments[0].kind = VERNON_PIPELINE_TENSOR;
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

    float output[4]{};
    ASSERT_EQ(vernonRhiDeviceDownloadBuffer(context.device, buffer.handle, 0, output, sizeof(output)),
              VERNON_RHI_STATUS_OK);
    for (int index = 0; index < 4; ++index)
        ASSERT_TRUE(output[index] == static_cast<float>(index) * factor);

    ASSERT_EQ(vernonRhiDeviceDestroyBuffer(context.device, buffer.handle), VERNON_RHI_STATUS_OK);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    vernonRuntimePipelineBundleDestroy(loaded);
    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
    vernonRhiDestroyDevice(context.device);
}
