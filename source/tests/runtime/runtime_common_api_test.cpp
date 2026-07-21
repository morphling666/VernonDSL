#include "vernon-c/Runtime.h"

#include <gtest/gtest.h>

#include <cstring>

TEST(RuntimeCommonApi, InitializesOptionsAndReportsErrors) {
  VernonRuntimeCreateOptions options{};
  VernonExternalOpenGLContext external{};
  VernonPipelineBundleLoadOptions bundleOptions{};
  VernonCpuInvocation invocation{};

  options.struct_size = sizeof(options);
  external.struct_size = sizeof(external);
  bundleOptions.struct_size = sizeof(bundleOptions);
  EXPECT_EQ(options.api_version_major, 0);
  EXPECT_EQ(options.api_version_minor, 0);
  EXPECT_EQ(options.reserved[0], 0);
  EXPECT_EQ(external.reserved[0], 0);
  EXPECT_EQ(bundleOptions.reserved[0], 0);
  EXPECT_EQ(invocation.arguments, nullptr);
  EXPECT_EQ(invocation.arguments_size, 0);
  EXPECT_EQ(VERNON_PIPELINE_INVOCATION_ABI_VERSION, 1);

  const VernonRuntimeCapabilities capabilities =
      vernonRuntimeGetCapabilities(VERNON_RUNTIME_CPU);
  EXPECT_TRUE(capabilities.available);
  EXPECT_TRUE(capabilities.supports_compute);
  EXPECT_TRUE(capabilities.supports_storage_buffers);

  VernonRuntimeContext *context =
      vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, &options);
  ASSERT_NE(context, nullptr);
  EXPECT_TRUE(vernonRuntimeGetContextCapabilities(context).available);
  EXPECT_EQ(vernonRuntimeGetLastError(context).size, 0);

  constexpr char malformedBundle[] = "{";
  constexpr char diagnosticPrefix[] = "invalid pipeline bundle:";
  bundleOptions.bundle_directory = ".";
  EXPECT_EQ(vernonRuntimeLoadPipelineBundleWithOptions(
                context, malformedBundle, sizeof(malformedBundle) - 1,
                &bundleOptions),
            nullptr);
  const VernonStringView diagnostic = vernonRuntimeGetLastError(context);
  ASSERT_GE(diagnostic.size, sizeof(diagnosticPrefix) - 1);
  EXPECT_EQ(std::strncmp(diagnostic.data, diagnosticPrefix,
                         sizeof(diagnosticPrefix) - 1),
            0);
  EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);

  context = vernonRuntimeCreate(VERNON_RUNTIME_CPU, 0);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(vernonRuntimeDestroy(context), VERNON_STATUS_OK);

  VernonRuntimeBackend target = VERNON_RUNTIME_CPU;
  EXPECT_EQ(vernonRuntimePipelineBundleInspectTarget(
                malformedBundle, sizeof(malformedBundle) - 1, &target),
            VERNON_STATUS_PARSE_ERROR);
}
