#include "vernon-c/Runtime.h"

#include <gtest/gtest.h>
#include <stdio.h>

#ifndef VERNON_COMPILER_CPU_BUNDLE_PATH
#error VERNON_COMPILER_CPU_BUNDLE_PATH must name the generated CPU bundle
#endif

TEST(RuntimeCpuAot, LoadsCompilerGeneratedBundle) {
    VernonRuntimeContext *runtime = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_TRUE(runtime);
    VernonLoadedPipeline *pipeline = vernonRuntimeLoadComputeBundle(runtime, VERNON_COMPILER_CPU_BUNDLE_PATH);
    if (!pipeline) {
        VernonStringView error = vernonRuntimeGetLastError(runtime);
        fprintf(stderr, "%.*s\n", (int)error.size, error.data);
    }
    ASSERT_TRUE(pipeline);
    vernonRuntimeLoadedPipelineDestroy(pipeline);
    ASSERT_TRUE(vernonRuntimeDestroy(runtime) == VERNON_STATUS_OK);
}
