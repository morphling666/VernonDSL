#include <vernon-c/Runtime.h>

#include <stddef.h>

void vernon_runtime_public_c_header_compile_check(void) {
    VernonRuntimeCreateOptions options = {0};
    VernonOpenGLContextCallbacks callbacks = {0};
    VernonProgramBundleLoadOptions bundle_options = {0};
    VernonCpuInvocation invocation = {0};
    VernonProgramArgument derivative_argument = {0};
    VernonAdDerivativeGroupView ad_group = {0};
    VernonProgramBindingToken binding_token = {0};
    VernonProgramResourceLease resource_lease = {0};
    VernonProgramBindingTelemetry binding_telemetry = {0};

    options.struct_size = sizeof(options);
    callbacks.struct_size = sizeof(callbacks);
    bundle_options.struct_size = sizeof(bundle_options);
    ad_group.struct_size = sizeof(ad_group);
    binding_token.struct_size = sizeof(binding_token);
    resource_lease.struct_size = sizeof(resource_lease);
    binding_telemetry.struct_size = sizeof(binding_telemetry);
    (void)options;
    (void)callbacks;
    (void)bundle_options;
    (void)invocation;
    (void)derivative_argument;
    (void)ad_group;
    (void)binding_token;
    (void)resource_lease;
    (void)binding_telemetry;
}
