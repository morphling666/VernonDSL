#include "VernonCompiler.h"
#include "VernonRuntime.h"

#include <cassert>
#include <cstring>

int main() {
  VernonRuntimeCapabilities capabilities =
      vernonRuntimeGetCapabilities(VERNON_RUNTIME_VULKAN);
  if (!capabilities.available)
    return 0;

  static constexpr char module[] = R"(
module {
  func.func @increment(
      %values: !vernon.buffer<f32, "read_write"> {
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
      name = "buffer_load"
    } : (!vernon.buffer<f32, "read_write">, index) -> f32
    %one = arith.constant 1.0 : f32
    %sum = arith.addf %value, %one : f32
    "vernon.intrinsic"(%values, %id, %sum) {
      name = "buffer_store"
    } : (!vernon.buffer<f32, "read_write">, index, f32) -> ()
    return
  }
}
)";

  VernonCompilerContext *compiler = vernonCompilerCreate();
  assert(compiler);
  VernonCompileResult *compiled = vernonCompilerCompileMlir(
      compiler, module, std::strlen(module), VERNON_TARGET_VULKAN);
  assert(compiled);
  assert(vernonCompileResultGetStatus(compiled) == VERNON_STATUS_OK);
  assert(vernonCompileResultGetArtifactCount(compiled) == 1);

  VernonRuntimeContext *runtime = vernonRuntimeCreate(VERNON_RUNTIME_VULKAN, 0);
  assert(runtime);
  VernonStringView artifact = vernonCompileResultGetArtifactData(compiled, 0);
  VernonStringView reflection = vernonCompileResultGetReflection(compiled);
  VernonLoadedKernel *kernel = vernonRuntimeLoadArtifact(
      runtime, artifact.data, artifact.size, reflection.data, reflection.size,
      "increment", std::strlen("increment"));
  assert(kernel);

  float input[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  float output[8] = {};
  VernonDeviceBuffer *buffer =
      vernonRuntimeBufferAllocate(runtime, sizeof(input), alignof(float));
  assert(buffer);
  assert(vernonRuntimeCopyFromHost(buffer, 0, input, sizeof(input)) ==
         VERNON_STATUS_OK);
  VernonLaunchArgument argument{VERNON_LAUNCH_TENSOR, buffer, nullptr, 0};
  assert(vernonRuntimeLaunch(kernel, {8, 1, 1}, &argument, 1) ==
         VERNON_STATUS_OK);
  assert(vernonRuntimeCopyToHost(buffer, 0, output, sizeof(output)) ==
         VERNON_STATUS_OK);
  for (int index = 0; index < 8; ++index)
    assert(output[index] == input[index] + 1.0f);

  assert(vernonRuntimeBufferFree(buffer) == VERNON_STATUS_OK);
  assert(vernonRuntimeKernelUnload(kernel) == VERNON_STATUS_OK);
  assert(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
  vernonCompileResultDestroy(compiled);
  vernonCompilerDestroy(compiler);
  return 0;
}
