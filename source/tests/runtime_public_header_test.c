#include <vernon-c/Runtime.h>

#include <assert.h>
#include <stddef.h>
#include <string.h>

int main(void) {
  VernonRuntimeCreateOptions options = {0};
  VernonExternalOpenGLContext external = {0};
  VernonPipelineBundleLoadOptions bundle_options = {0};
  VernonCpuInvocation invocation = {0};

  options.struct_size = sizeof(options);
  external.struct_size = sizeof(external);
  bundle_options.struct_size = sizeof(bundle_options);

  assert(options.api_version_major == 0 && options.api_version_minor == 0);
  assert(options.reserved[0] == 0 && external.reserved[0] == 0);
  assert(bundle_options.reserved[0] == 0);
  assert(invocation.arguments == NULL && invocation.arguments_size == 0);
  assert(VERNON_PIPELINE_INVOCATION_ABI_VERSION == 1);

  VernonRuntimeCapabilities capabilities =
      vernonRuntimeGetCapabilities(VERNON_RUNTIME_CPU);
  assert(capabilities.available && capabilities.supports_compute);
  assert(capabilities.supports_storage_buffers);

  VernonRuntimeContext *context =
      vernonRuntimeCreateWithOptions(VERNON_RUNTIME_CPU, &options);
  assert(context != NULL);
  capabilities = vernonRuntimeGetContextCapabilities(context);
  assert(capabilities.available && capabilities.supports_compute);
  assert(vernonRuntimeGetLastError(context).size == 0);

  {
    static const char malformed_bundle[] = "{";
    static const char diagnostic_prefix[] = "invalid pipeline bundle:";
    bundle_options.bundle_directory = ".";
    assert(vernonRuntimeLoadPipelineBundleWithOptions(
               context, malformed_bundle, sizeof(malformed_bundle) - 1,
               &bundle_options) == NULL);
    VernonStringView diagnostic = vernonRuntimeGetLastError(context);
    assert(diagnostic.size >= sizeof(diagnostic_prefix) - 1);
    assert(strncmp(diagnostic.data, diagnostic_prefix,
                   sizeof(diagnostic_prefix) - 1) == 0);
  }
  assert(vernonRuntimeDestroy(context) == VERNON_STATUS_OK);

  /* Keep the compatibility constructor covered by the public C ABI test. */
  context = vernonRuntimeCreate(VERNON_RUNTIME_CPU, 0);
  assert(context != NULL);
  assert(vernonRuntimeDestroy(context) == VERNON_STATUS_OK);

  {
    static const char malformed_bundle[] = "{";
    VernonRuntimeBackend target = VERNON_RUNTIME_CPU;
    assert(vernonRuntimePipelineBundleInspectTarget(
               malformed_bundle, sizeof(malformed_bundle) - 1, &target) ==
           VERNON_STATUS_PARSE_ERROR);
  }
  return 0;
}
