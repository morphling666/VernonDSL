#include "native_runtime.h"

RhiHostState *runtimeRhiHost(const Runtime *runtime) { return runtime ? runtime->rhiHost.get() : nullptr; }

std::unique_ptr<Runtime> createRhiRuntime(RhiHost &host) {
    VernonRuntimeBackend backend;
    switch (host.state->backend) {
    case VERNON_RHI_BACKEND_CUDA:
        backend = VERNON_RUNTIME_CUDA;
        break;
    case VERNON_RHI_BACKEND_VULKAN:
        backend = VERNON_RUNTIME_VULKAN;
        break;
    case VERNON_RHI_BACKEND_DIRECTX12:
        backend = VERNON_RUNTIME_DIRECTX12;
        break;
    case VERNON_RHI_BACKEND_METAL:
        backend = VERNON_RUNTIME_METAL;
        break;
    case VERNON_RHI_BACKEND_OPENGL:
        backend = VERNON_RUNTIME_OPENGL;
        break;
    case VERNON_RHI_BACKEND_OPENGL_ES:
        backend = VERNON_RUNTIME_OPENGL_ES;
        break;
    }
    VernonRuntimeContext *runtime = vernonRuntimeCreateForRhiDevice(backend, host.state->device);
    if (!runtime)
        throw std::runtime_error("cannot create Runtime for Vernon RHI device");
    return std::make_unique<Runtime>(runtime, host.state);
}
