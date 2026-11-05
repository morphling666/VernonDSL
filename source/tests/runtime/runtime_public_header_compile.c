#include <VernonHal.h>
#include <VernonSchedule.h>
#include <vernon-c/Runtime.h>

#include <stddef.h>

void vernon_hal_public_c_header_compile_check(void) {
    VernonHalDeviceDescriptor device = {0};
    VernonHalBufferDescriptor buffer = {0};
    VernonHalImageRegion region = {0};
    VernonHalSamplerDescriptor sampler = {0};
    VernonHalSurfaceDescriptor surface = {0};
    VernonHalSurfaceCapabilities capabilities = {0};
    VernonHalSurfaceConfiguration configuration = {0};
    VernonHalSurfaceImage image = {0};
    (void)device;
    (void)buffer;
    (void)region;
    (void)sampler;
    (void)surface;
    (void)capabilities;
    (void)configuration;
    (void)image;
    (void)&vernonHalCreateDevice;
    (void)&vernonHalSurfacePresent;
}

void vernon_runtime_public_c_header_compile_check(void) {
    VernonRuntimeCreateOptions options = {0};
    VernonOpenGLContextCallbacks callbacks = {0};
    VernonProgramBundleLoadOptions bundle_options = {0};
    VernonCpuInvocation invocation = {0};
    VernonProgramArgument derivative_argument = {0};
    VernonAdDerivativeGroupView ad_group = {0};

    options.struct_size = sizeof(options);
    callbacks.struct_size = sizeof(callbacks);
    bundle_options.struct_size = sizeof(bundle_options);
    ad_group.struct_size = sizeof(ad_group);
    (void)options;
    (void)callbacks;
    (void)bundle_options;
    (void)invocation;
    (void)derivative_argument;
    (void)ad_group;
}
