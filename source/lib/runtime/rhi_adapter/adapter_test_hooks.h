#ifndef VERNON_RUNTIME_RHI_ADAPTER_TEST_HOOKS_H
#define VERNON_RUNTIME_RHI_ADAPTER_TEST_HOOKS_H

#include "VernonRuntimeRHIAdapter.h"

#include <cstddef>

namespace vernon::runtime {

struct RhiAdapterPreparationStats {
    size_t shaderPreparations{};
    size_t layoutPreparations{};
    size_t pipelinePreparations{};
    size_t bindingCreations{};
    size_t bindingSnapshotCreations{};
    size_t dispatches{};
};

RhiAdapterPreparationStats getRhiAdapterPreparationStats(const VernonRuntimeRhiAdapter &adapter);

} // namespace vernon::runtime

#endif
