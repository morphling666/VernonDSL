#include "vernon-c/Runtime.h"

#include <gtest/gtest.h>
#include <stdio.h>

#ifndef VERNON_COMPILER_CPU_BUNDLE_PATH
#error VERNON_COMPILER_CPU_BUNDLE_PATH must name the generated CPU bundle
#endif

TEST(RuntimeCpuAot, LoadsCompilerGeneratedBundle) {
  VernonRuntimeContext *runtime = vernonRuntimeCreate(VERNON_RUNTIME_CPU, 0);
  ASSERT_TRUE(runtime);
  VernonLoadedKernel *kernel =
      vernonRuntimeLoadComputeBundle(runtime, VERNON_COMPILER_CPU_BUNDLE_PATH);
  if (!kernel) {
    VernonStringView error = vernonRuntimeGetLastError(runtime);
    fprintf(stderr, "%.*s\n", (int)error.size, error.data);
  }
  ASSERT_TRUE(kernel);
  ASSERT_TRUE(vernonRuntimeKernelUnload(kernel) == VERNON_STATUS_OK);
  ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
}
