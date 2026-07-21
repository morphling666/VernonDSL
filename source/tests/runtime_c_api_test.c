#include "vernon-c/Runtime.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef VERNON_CPU_BUNDLE_PATH
#error VERNON_CPU_BUNDLE_PATH must name the CPU bundle test fixture
#endif

static VernonStatus fill_grid(const VernonCpuInvocation *invocation) {
  uintptr_t address = 0;
  uint32_t gid[3] = {0, 0, 0};
  memcpy(&address, invocation->arguments, sizeof(address));
  memcpy(gid, (const unsigned char *)invocation->arguments + 8, sizeof(gid));
  float *values = (float *)address;
  values[gid[2] * 6 + gid[1] * 3 + gid[0]] =
      (float)(gid[0] + 10 * gid[1] + 100 * gid[2]);
  return VERNON_STATUS_OK;
}

int main(void) {
  static const char static_symbol[] = "__vernon_cpu_test_fill";
  assert(vernonRuntimeRegisterStaticCpuEntry(
             (VernonStringView){static_symbol, sizeof(static_symbol) - 1},
             fill_grid) == VERNON_STATUS_OK);
  assert(vernonRuntimeRegisterStaticCpuEntry(
             (VernonStringView){static_symbol, sizeof(static_symbol) - 1},
             fill_grid) == VERNON_STATUS_OK);
  assert(vernonRuntimeRegisterStaticCpuEntry((VernonStringView){NULL, 0},
                                             fill_grid) ==
         VERNON_STATUS_INVALID_ARGUMENT);
  static const char reflection[] =
      "{\"gpu_launch_abi_version\":1,\"entries\":[{\"name\":\"fill\","
      "\"cpu_arguments_size\":20,\"workgroup_size\":[2,2,1],\"arguments\":["
      "{\"kind\":\"tensor\",\"cpu_offset\":0,\"cpu_size\":8},"
      "{\"kind\":\"builtin\",\"builtin\":\"global_invocation_id\","
      "\"cpu_offset\":8,\"cpu_size\":12}]}]}";
  VernonRuntimeCapabilities capabilities =
      vernonRuntimeGetCapabilities(VERNON_RUNTIME_CPU);
  assert(capabilities.available && capabilities.supports_compute);
  VernonRuntimeContext *runtime = vernonRuntimeCreate(VERNON_RUNTIME_CPU, 0);
  assert(runtime);
  VernonDeviceBuffer *buffer =
      vernonRuntimeBufferAllocate(runtime, 12 * sizeof(float), 4);
  assert(buffer);
  static const char bad_reflection[] =
      "{\"gpu_launch_abi_version\":2,\"entries\":[]}";
  assert(!vernonRuntimeLoadCpuEntry(runtime, fill_grid, bad_reflection,
                                    sizeof(bad_reflection) - 1, "fill", 4));
  VernonLoadedKernel *kernel = vernonRuntimeLoadCpuEntry(
      runtime, fill_grid, reflection, sizeof(reflection) - 1, "fill", 4);
  assert(kernel);
  VernonLaunchArgument argument = {VERNON_LAUNCH_TENSOR, buffer, NULL, 0};
  VernonLaunchSize grid = {3, 2, 2};
  assert(vernonRuntimeLaunch(kernel, grid, &argument, 1) == VERNON_STATUS_OK);
  float values[12] = {0};
  assert(vernonRuntimeCopyToHost(buffer, 0, values, sizeof(values)) ==
         VERNON_STATUS_OK);
  assert(values[0] == 0.0f && values[2] == 2.0f);
  assert(values[3] == 10.0f && values[11] == 112.0f);
  assert(vernonRuntimeDestroy(runtime) == VERNON_STATUS_INVALID_ARGUMENT);
  assert(vernonRuntimeKernelUnload(kernel) == VERNON_STATUS_OK);

  memset(values, 0, sizeof(values));
  assert(vernonRuntimeCopyFromHost(buffer, 0, values, sizeof(values)) ==
         VERNON_STATUS_OK);
  kernel = vernonRuntimeLoadComputeBundle(runtime, VERNON_CPU_BUNDLE_PATH);
  if (!kernel) {
    VernonStringView error = vernonRuntimeGetLastError(runtime);
    fprintf(stderr, "%.*s\n", (int)error.size, error.data);
  }
  assert(kernel);
  assert(vernonRuntimeLaunch(kernel, grid, &argument, 1) == VERNON_STATUS_OK);
  assert(vernonRuntimeCopyToHost(buffer, 0, values, sizeof(values)) ==
         VERNON_STATUS_OK);
  assert(values[0] == 0.0f && values[2] == 2.0f);
  assert(values[3] == 10.0f && values[11] == 112.0f);
  assert(vernonRuntimeKernelUnload(kernel) == VERNON_STATUS_OK);

  assert(vernonRuntimeBufferFree(buffer) == VERNON_STATUS_OK);
  assert(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
  return 0;
}
