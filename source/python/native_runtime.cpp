#include "native_runtime.h"

#include "native_lifecycle_test_hooks.h"

#include <atomic>

namespace {

#if defined(VERNON_ENABLE_LIFECYCLE_TEST_HOOKS)
std::atomic<uint64_t> destroyedRuntimes{};
std::atomic<uint64_t> destroyedRhiDevices{};
std::atomic<uint64_t> destroyedOwnedContexts{};
std::atomic<uint64_t> destructionSequence{};
std::atomic<uint64_t> runtimeDestructionOrder{};
std::atomic<uint64_t> rhiDeviceDestructionOrder{};
std::atomic<uint64_t> ownedContextDestructionOrder{};
#endif

} // namespace

namespace vernon::python::testing {

void noteRuntimeDestroyed() noexcept {
#if defined(VERNON_ENABLE_LIFECYCLE_TEST_HOOKS)
    destroyedRuntimes.fetch_add(1, std::memory_order_relaxed);
    runtimeDestructionOrder.store(destructionSequence.fetch_add(1, std::memory_order_relaxed) + 1,
                                  std::memory_order_relaxed);
#endif
}

void noteRhiDeviceDestroyed() noexcept {
#if defined(VERNON_ENABLE_LIFECYCLE_TEST_HOOKS)
    destroyedRhiDevices.fetch_add(1, std::memory_order_relaxed);
    rhiDeviceDestructionOrder.store(destructionSequence.fetch_add(1, std::memory_order_relaxed) + 1,
                                    std::memory_order_relaxed);
#endif
}

void noteOwnedContextDestroyed() noexcept {
#if defined(VERNON_ENABLE_LIFECYCLE_TEST_HOOKS)
    destroyedOwnedContexts.fetch_add(1, std::memory_order_relaxed);
    ownedContextDestructionOrder.store(destructionSequence.fetch_add(1, std::memory_order_relaxed) + 1,
                                       std::memory_order_relaxed);
#endif
}

void resetLifecycleCounts() noexcept {
#if defined(VERNON_ENABLE_LIFECYCLE_TEST_HOOKS)
    destroyedRuntimes.store(0, std::memory_order_relaxed);
    destroyedRhiDevices.store(0, std::memory_order_relaxed);
    destroyedOwnedContexts.store(0, std::memory_order_relaxed);
    destructionSequence.store(0, std::memory_order_relaxed);
    runtimeDestructionOrder.store(0, std::memory_order_relaxed);
    rhiDeviceDestructionOrder.store(0, std::memory_order_relaxed);
    ownedContextDestructionOrder.store(0, std::memory_order_relaxed);
#endif
}

LifecycleCounts lifecycleCounts() noexcept {
#if defined(VERNON_ENABLE_LIFECYCLE_TEST_HOOKS)
    return {
        destroyedRuntimes.load(std::memory_order_relaxed),
        destroyedRhiDevices.load(std::memory_order_relaxed),
        destroyedOwnedContexts.load(std::memory_order_relaxed),
        runtimeDestructionOrder.load(std::memory_order_relaxed),
        rhiDeviceDestructionOrder.load(std::memory_order_relaxed),
        ownedContextDestructionOrder.load(std::memory_order_relaxed),
    };
#else
    return {};
#endif
}

} // namespace vernon::python::testing

PythonProgramExecutable::~PythonProgramExecutable() {
    vernonRuntimeProgramExecutableDestroy(executable);
    for (size_t index = 0; index < registeredCpuEntries.size(); ++index) {
        const auto &[symbol, entry] = registeredCpuEntries[index];
        vernonRuntimeUnregisterCpuEntry(runtime, {symbol.data(), symbol.size()}, entry);
        if (index < internedCpuJits.size())
            owner->releaseInternedCpuJit(symbol, internedCpuJits[index]);
    }
    internedCpuJits.clear();
    vernonRuntimeProgramBundleDestroy(bundle);
}

RhiHostState *runtimeRhiHost(const RuntimeState *runtime) { return runtime ? runtime->rhiHost.get() : nullptr; }

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
    RuntimeContextOwner runtime(vernonRuntimeCreateForRhiDevice(backend, host.state->device));
    if (!runtime)
        throw std::runtime_error("cannot create Runtime for Vernon RHI device");
    return std::make_unique<Runtime>(std::move(runtime), host.state);
}
