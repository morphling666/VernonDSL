#include "vernon-c/Runtime.h"

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
  static constexpr char reflection[] = R"({
    "gpu_launch_abi_version": 1,
    "entries": [{
      "name": "scale",
      "workgroup_size": [4, 1, 1],
      "cpu_arguments_size": 12,
      "arguments": [
        {"kind": "tensor", "dtype": "f32", "shape": [4],
         "alignment": 4, "cpu_offset": 0, "cpu_size": 8},
        {"kind": "scalar", "dtype": "f32", "alignment": 4,
         "cpu_offset": 8, "cpu_size": 4}
      ]
    }]
  })";
  const std::string bundle =
      R"({"pipeline_bundle_schema_version":1,"invocation_abi_version":1,)"
      R"("type":"vernon_pipeline_bundle","id":"cuda/scale","target":"cuda",)"
      R"("features":[],"variants":[{"key":[],"parameters":[)"
      R"({"slot":0,"name":"output","kind":"tensor","dtype":"f32",)"
      R"("shape":[4],"access":"write","uses":[{"stage":"compute",)"
      R"("entry":"scale","index":0,"kind":"tensor","dtype":"f32",)"
      R"("shape":[4],"interface":"storage","access":"write"}]},)"
      R"({"slot":1,"name":"factor","kind":"inline","dtype":"f32",)"
      R"("shape":[],"access":"read","uses":[{"stage":"compute",)"
      R"("entry":"scale","index":1,"kind":"scalar","dtype":"f32",)"
      R"("shape":[],"interface":"value","access":"read"}]})"
      R"(],"outputs":[{"name":"result","kind":"tensor","dtype":"f32",)"
      R"("shape":[4],"access":"write","location":0}],)"
      R"("steps":[{"kind":"dispatch","stage":"scale"}]}],)"
      R"("stage_artifacts":{"scale":{"id":"scale","entry":"scale",)"
      R"("stage":"compute","target":"cuda","format":"ptx","source":)" +
      jsonString(ptx) + R"(,"reflection":)" + reflection + "}}}";

  VernonRuntimeBackend target = VERNON_RUNTIME_CPU;
  ASSERT_TRUE(vernonRuntimePipelineBundleInspectTarget(
                  bundle.data(), bundle.size(), &target) == VERNON_STATUS_OK);
  ASSERT_TRUE(target == VERNON_RUNTIME_CUDA);

  if (!vernonRuntimeGetCapabilities(VERNON_RUNTIME_CUDA).available) {
    GTEST_SKIP() << "CUDA runtime backend is unavailable";
  }

  VernonRuntimeContext *runtime = vernonRuntimeCreate(VERNON_RUNTIME_CUDA, 0);
  ASSERT_TRUE(runtime);
  std::string barrierBundle = bundle;
  const std::string dispatchStep = R"({"kind":"dispatch","stage":"scale"}])";
  const size_t dispatch = barrierBundle.find(dispatchStep);
  ASSERT_TRUE(dispatch != std::string::npos);
  barrierBundle.replace(
      dispatch, dispatchStep.size(),
      R"({"kind":"dispatch","stage":"scale"},{"kind":"barrier"})");
  ASSERT_TRUE(!vernonRuntimeLoadPipelineBundle(runtime, barrierBundle.data(),
                                               barrierBundle.size()));

  VernonPipelineBundle *loaded =
      vernonRuntimeLoadPipelineBundle(runtime, bundle.data(), bundle.size());
  ASSERT_TRUE(loaded);
  VernonLoadedPipeline *pipeline =
      vernonRuntimeResolvePipeline(loaded, {nullptr, 0});
  ASSERT_TRUE(pipeline);

  ASSERT_TRUE(vernonRuntimeLoadedPipelineGetParameterCount(pipeline) == 2);
  VernonPipelineParameterView parameter{};
  ASSERT_TRUE(vernonRuntimeLoadedPipelineGetParameterByIndex(
                  pipeline, 0, &parameter) == VERNON_STATUS_OK);
  ASSERT_TRUE(parameter.slot == 0 && parameter.kind == VERNON_PIPELINE_TENSOR &&
              parameter.dtype == VERNON_DATA_F32 &&
              parameter.access == VERNON_ACCESS_WRITE && parameter.rank == 1 &&
              parameter.static_shape[0] == 4);
  ASSERT_TRUE(vernonRuntimeLoadedPipelineFindParameter(
                  pipeline, {"factor", std::strlen("factor")}, &parameter) ==
              VERNON_STATUS_OK);
  ASSERT_TRUE(parameter.slot == 1 &&
              parameter.kind == VERNON_PIPELINE_INLINE_VALUE);
  ASSERT_TRUE(vernonRuntimeLoadedPipelineGetOutputCount(pipeline) == 1);
  VernonPipelineOutputView pipelineOutput{};
  ASSERT_TRUE(vernonRuntimeLoadedPipelineFindOutput(
                  pipeline, {"result", std::strlen("result")},
                  &pipelineOutput) == VERNON_STATUS_OK);
  ASSERT_TRUE(pipelineOutput.kind == VERNON_PIPELINE_TENSOR &&
              pipelineOutput.dtype == VERNON_DATA_F32 &&
              pipelineOutput.access == VERNON_ACCESS_WRITE &&
              pipelineOutput.rank == 1 && pipelineOutput.static_shape[0] == 4 &&
              pipelineOutput.location == 0);

  VernonDeviceBuffer *buffer =
      vernonRuntimeBufferAllocate(runtime, 4 * sizeof(float), alignof(float));
  ASSERT_TRUE(buffer);
  const uint64_t shape[] = {4};
  const uint64_t strides[] = {sizeof(float)};
  const float factor = 3.0f;
  VernonPipelineArgument arguments[2]{};
  arguments[0].slot = 0;
  arguments[0].kind = VERNON_PIPELINE_TENSOR;
  arguments[0].tensor = {
      buffer, VERNON_DATA_F32, VERNON_ACCESS_WRITE, 1, shape, strides, 0};
  arguments[1].slot = 1;
  arguments[1].kind = VERNON_PIPELINE_INLINE_VALUE;
  arguments[1].inline_value = {VERNON_DATA_F32, 0, nullptr, &factor,
                               sizeof(factor)};
  VernonPipelineInvocation invocation{};
  invocation.struct_size = sizeof(invocation);
  invocation.abi_version = VERNON_PIPELINE_INVOCATION_ABI_VERSION;
  invocation.arguments = arguments;
  invocation.argument_count = 2;
  invocation.compute_grid = {4, 1, 1};
  ASSERT_TRUE(vernonRuntimePipelineInvoke(pipeline, &invocation) ==
              VERNON_STATUS_OK);
  ASSERT_TRUE(vernonRuntimeSynchronize(runtime) == VERNON_STATUS_OK);

  float output[4]{};
  ASSERT_TRUE(vernonRuntimeCopyToHost(buffer, 0, output, sizeof(output)) ==
              VERNON_STATUS_OK);
  for (int index = 0; index < 4; ++index)
    ASSERT_TRUE(output[index] == static_cast<float>(index) * factor);

  ASSERT_TRUE(vernonRuntimeBufferFree(buffer) == VERNON_STATUS_OK);
  vernonRuntimeLoadedPipelineDestroy(pipeline);
  vernonRuntimePipelineBundleDestroy(loaded);
  ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
}
