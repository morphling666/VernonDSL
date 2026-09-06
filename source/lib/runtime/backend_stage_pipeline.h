#ifndef VERNON_RUNTIME_BACKEND_STAGE_PIPELINE_H
#define VERNON_RUNTIME_BACKEND_STAGE_PIPELINE_H

#include "VernonRuntime.h"
#include "pipeline_bundle.h"
#include "pipeline_manifest.h"
#include "pipeline_metadata.h"

#include <string>
#include <unordered_map>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::runtime {

// Construction-only backend input. It is not a deployment identity and never
// owns canonical Program or Node state.
struct BackendPipelineBundle {
    VernonRuntimeContext *context{};
    std::unordered_map<std::string, Stage> stages;
};

} // namespace vernon::runtime

// Internal definition of the public opaque C handle. The projection is only
// the immutable ABI of one reusable physical Stage implementation; canonical
// Program Value bindings and Node controls live in ResolvedNodePlan.
struct VernonStageExecutable {
    VernonStageExecutable() = default;
    VernonStageExecutable(const VernonStageExecutable &) = delete;
    VernonStageExecutable &operator=(const VernonStageExecutable &) = delete;
    VernonStageExecutable(VernonStageExecutable &&) = delete;
    VernonStageExecutable &operator=(VernonStageExecutable &&) = delete;
    ~VernonStageExecutable();

    VernonRuntimeContext *context{};
    vernon::runtime::Variant bindingProjection;
    VernonLaunchSize workgroupSize{1, 1, 1};
    vernon::runtime::DispatchContract dispatchContract;
    std::vector<vernon::runtime::TensorViewWriteFootprint> readFootprints;
    std::vector<vernon::runtime::TensorViewWriteFootprint> writeFootprints;
    void *backendState{};
    void (*destroyBackendState)(void *){};
};

#endif
