#include "vernon-c/Runtime.h"

#include <assert.h>
#include <stdio.h>

#ifndef VERNON_COMPILER_CPU_BUNDLE_PATH
#error VERNON_COMPILER_CPU_BUNDLE_PATH must name the generated CPU bundle
#endif

int main(void) {
  VernonRuntimeContext *runtime = vernonRuntimeCreate(VERNON_RUNTIME_CPU, 0);
  assert(runtime);
  VernonLoadedKernel *kernel =
      vernonRuntimeLoadComputeBundle(runtime, VERNON_COMPILER_CPU_BUNDLE_PATH);
  if (!kernel) {
    VernonStringView error = vernonRuntimeGetLastError(runtime);
    fprintf(stderr, "%.*s\n", (int)error.size, error.data);
  }
  assert(kernel);
  assert(vernonRuntimeKernelUnload(kernel) == VERNON_STATUS_OK);
  assert(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
  return 0;
}
