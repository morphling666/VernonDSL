#include "vernon-c/Runtime.h"

#include <cassert>
#include <cstring>

int main() {
  VernonRuntimeCapabilities capabilities =
      vernonRuntimeGetCapabilities(VERNON_RUNTIME_CUDA);
  if (!capabilities.available)
    return 0;
  VernonRuntimeContext *runtime = vernonRuntimeCreate(VERNON_RUNTIME_CUDA, 0);
  assert(runtime);
  VernonDeviceBuffer *buffer =
      vernonRuntimeBufferAllocate(runtime, 16, alignof(float));
  assert(buffer);
  float input[4] = {1, 2, 3, 4};
  float output[4] = {};
  assert(vernonRuntimeCopyFromHost(buffer, 0, input, sizeof(input)) ==
         VERNON_STATUS_OK);
  assert(vernonRuntimeCopyToHost(buffer, 0, output, sizeof(output)) ==
         VERNON_STATUS_OK);
  for (int index = 0; index < 4; ++index)
    assert(input[index] == output[index]);

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
  VernonLoadedKernel *kernel =
      vernonRuntimeLoadArtifact(runtime, ptx, std::strlen(ptx), reflection,
                                std::strlen(reflection), "scale", 5);
  assert(kernel);
  const float factor = 2.0f;
  VernonLaunchArgument arguments[] = {
      {VERNON_LAUNCH_TENSOR, buffer, nullptr, 0},
      {VERNON_LAUNCH_SCALAR, nullptr, &factor, sizeof(factor)},
  };
  assert(vernonRuntimeLaunch(kernel, {4, 1, 1}, arguments, 2) ==
         VERNON_STATUS_OK);
  assert(vernonRuntimeSynchronize(runtime) == VERNON_STATUS_OK);
  assert(vernonRuntimeCopyToHost(buffer, 0, output, sizeof(output)) ==
         VERNON_STATUS_OK);
  for (int index = 0; index < 4; ++index)
    assert(output[index] == static_cast<float>(index) * factor);
  assert(vernonRuntimeKernelUnload(kernel) == VERNON_STATUS_OK);

  assert(vernonRuntimeBufferFree(buffer) == VERNON_STATUS_OK);
  assert(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
  return 0;
}
