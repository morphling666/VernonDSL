#ifndef VERNON_RUNTIME_BACKEND_STAGE_PIPELINE_H
#define VERNON_RUNTIME_BACKEND_STAGE_PIPELINE_H

#include "VernonRuntime.h"
#include "graphics_invocation_planner.h"
#include "pipeline_metadata.h"
#include "resolved_stage_types.h"
#include "runtime_lifecycle.h"
#include "stage_artifact.h"
#include "stage_binding_plan.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct VernonRuntimeContext;

namespace vernon::schedule {
struct Fragment;
} // namespace vernon::schedule

namespace vernon::runtime {

// Construction-only backend input. It is not a deployment identity and never
// owns canonical Program or Node state.
struct BackendStageBuildInputs {
    VernonRuntimeContext *context{};
    std::unordered_map<std::string, LoadedStageArtifact> artifacts;
};

enum class BackendPipelineError : uint8_t {
    InvalidBindingPlan,
    UnsupportedBackend,
    BackendResolutionFailed,
    LifecycleUnavailable,
    InvalidArtifact,
};

struct BackendPipelineFailure {
    BackendPipelineError code;
    std::string diagnostic;

    BackendPipelineFailure(BackendPipelineError value, std::string detail = {})
        : code(value), diagnostic(std::move(detail)) {}
};

using BackendPipelineResult = vernon::Result<void, BackendPipelineFailure>;
std::string renderBackendPipelineError(const BackendPipelineFailure &error);

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

    vernon::Option<vernon::runtime::RuntimeChildLifecycle> lifecycle;
    VernonRuntimeContext *context{};
    vernon::runtime::StageBindingPlan bindingProjection;
    // Graphics Stages only.
    vernon::runtime::GraphicsDrawLayout drawLayout;
    VernonLaunchSize workgroupSize{1, 1, 1};
    vernon::runtime::DispatchContract dispatchContract;
    std::vector<vernon::runtime::TensorViewWriteFootprint> readFootprints;
    std::vector<vernon::runtime::TensorViewWriteFootprint> writeFootprints;
    // A CPU or CUDA compute Stage as the one-kernel fragment a linked schedule runs it with; null with the reason
    // when the Stage uses what fragments do not carry.
    std::shared_ptr<const vernon::schedule::Fragment> kernelFragment;
    std::string kernelFragmentError;
    void *backendState{};
    void (*destroyBackendState)(void *){};
};

#endif
