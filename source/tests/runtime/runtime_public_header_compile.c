#include <vernon-c/Runtime.h>

#include <stddef.h>

void vernon_runtime_public_c_header_compile_check(void) {
  VernonRuntimeCreateOptions options = {0};
  VernonExternalOpenGLContext external = {0};
  VernonPipelineBundleLoadOptions bundle_options = {0};
  VernonCpuInvocation invocation = {0};

  options.struct_size = sizeof(options);
  external.struct_size = sizeof(external);
  bundle_options.struct_size = sizeof(bundle_options);
  (void)options;
  (void)external;
  (void)bundle_options;
  (void)invocation;
}
