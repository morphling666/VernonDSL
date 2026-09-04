#include "vernon-c/Runtime.h"

#include <gtest/gtest.h>

#include <cstring>

TEST(RuntimeCommonApi, InitializesOptionsAndReportsErrors) {
    VernonRuntimeCreateOptions options{};
    VernonOpenGLContextCallbacks callbacks{};
    VernonProgramBundleLoadOptions bundleOptions{};
    VernonCpuInvocation invocation{};

    options.struct_size = sizeof(options);
    callbacks.struct_size = sizeof(callbacks);
    bundleOptions.struct_size = sizeof(bundleOptions);
    EXPECT_EQ(options.reserved[0], 0);
    EXPECT_EQ(callbacks.reserved[0], 0);
    EXPECT_EQ(bundleOptions.reserved[0], 0);
    EXPECT_EQ(invocation.arguments, nullptr);
    EXPECT_EQ(invocation.arguments_size, 0);

    const VernonRuntimeCapabilities capabilities = vernonRuntimeGetCapabilities(VERNON_RUNTIME_CPU);
    EXPECT_TRUE(capabilities.available);
    EXPECT_TRUE(capabilities.supports_compute);
    EXPECT_TRUE(capabilities.supports_storage_buffers);

    VernonRuntimeContext *context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, &options);
    ASSERT_NE(context, nullptr);
    EXPECT_TRUE(vernonRuntimeGetContextCapabilities(context).available);
    EXPECT_EQ(vernonRuntimeGetLastError(context).size, 0);

    constexpr char malformedBundle[] = "{";
    constexpr char diagnosticPrefix[] = "invalid pipeline bundle:";
    bundleOptions.bundle_directory = ".";
    EXPECT_EQ(vernonRuntimeLoadProgramBundleWithOptions(context, malformedBundle, sizeof(malformedBundle) - 1,
                                                        &bundleOptions),
              nullptr);
    const VernonStringView diagnostic = vernonRuntimeGetLastError(context);
    ASSERT_GE(diagnostic.size, sizeof(diagnosticPrefix) - 1);
    EXPECT_EQ(std::strncmp(diagnostic.data, diagnosticPrefix, sizeof(diagnosticPrefix) - 1), 0);
    const VernonRuntimeCapabilities refreshed = vernonRuntimeGetContextCapabilities(context);
    EXPECT_TRUE(refreshed.available);
    EXPECT_EQ(refreshed.diagnostic.size, 0);
    EXPECT_EQ(vernonRuntimeGetLastError(context).size, 0);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);

    context = vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, nullptr);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);

    VernonRuntimeBackend target = VERNON_RUNTIME_CPU;
    EXPECT_EQ(vernonRuntimeProgramBundleInspectTarget(malformedBundle, sizeof(malformedBundle) - 1, &target),
              VERNON_STATUS_PARSE_ERROR);
}
