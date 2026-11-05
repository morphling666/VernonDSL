#include "native_runtime.h"

#include "native_lifecycle_test_hooks.h"

#include <atomic>

namespace {

#if defined(VERNON_ENABLE_LIFECYCLE_TEST_HOOKS)
std::atomic<uint64_t> destroyedRuntimes{};
std::atomic<uint64_t> destroyedDevices{};
std::atomic<uint64_t> destroyedOwnedContexts{};
std::atomic<uint64_t> destructionSequence{};
std::atomic<uint64_t> runtimeDestructionOrder{};
std::atomic<uint64_t> deviceDestructionOrder{};
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

void noteDeviceDestroyed() noexcept {
#if defined(VERNON_ENABLE_LIFECYCLE_TEST_HOOKS)
    destroyedDevices.fetch_add(1, std::memory_order_relaxed);
    deviceDestructionOrder.store(destructionSequence.fetch_add(1, std::memory_order_relaxed) + 1,
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
    destroyedDevices.store(0, std::memory_order_relaxed);
    destroyedOwnedContexts.store(0, std::memory_order_relaxed);
    destructionSequence.store(0, std::memory_order_relaxed);
    runtimeDestructionOrder.store(0, std::memory_order_relaxed);
    deviceDestructionOrder.store(0, std::memory_order_relaxed);
    ownedContextDestructionOrder.store(0, std::memory_order_relaxed);
#endif
}

LifecycleCounts lifecycleCounts() noexcept {
#if defined(VERNON_ENABLE_LIFECYCLE_TEST_HOOKS)
    return {
        destroyedRuntimes.load(std::memory_order_relaxed),
        destroyedDevices.load(std::memory_order_relaxed),
        destroyedOwnedContexts.load(std::memory_order_relaxed),
        runtimeDestructionOrder.load(std::memory_order_relaxed),
        deviceDestructionOrder.load(std::memory_order_relaxed),
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

bool runtimeOnDevice(const RuntimeState *runtime) { return runtime && runtime->device->backend != VERNON_RUNTIME_CPU; }

std::unique_ptr<Runtime> deviceRuntime(HalDevice &device) {
    std::shared_ptr<RuntimeState> state = device.state->runtime.lock();
    if (!state) {
        state = std::make_shared<RuntimeState>(device.state);
        device.state->runtime = state;
    }
    return std::make_unique<Runtime>(std::move(state));
}
