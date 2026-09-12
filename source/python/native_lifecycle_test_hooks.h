#ifndef VERNON_PYTHON_NATIVE_LIFECYCLE_TEST_HOOKS_H
#define VERNON_PYTHON_NATIVE_LIFECYCLE_TEST_HOOKS_H

#include <cstdint>

namespace vernon::python::testing {

struct LifecycleCounts {
    uint64_t runtimes{};
    uint64_t rhiDevices{};
    uint64_t ownedContexts{};
    uint64_t runtimeOrder{};
    uint64_t rhiDeviceOrder{};
    uint64_t ownedContextOrder{};
};

void noteRuntimeDestroyed() noexcept;
void noteRhiDeviceDestroyed() noexcept;
void noteOwnedContextDestroyed() noexcept;
void resetLifecycleCounts() noexcept;
LifecycleCounts lifecycleCounts() noexcept;

} // namespace vernon::python::testing

#endif
