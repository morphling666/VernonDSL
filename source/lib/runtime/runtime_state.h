#ifndef VERNON_RUNTIME_RUNTIME_STATE_H
#define VERNON_RUNTIME_RUNTIME_STATE_H

#include "VernonRuntime.h"
#include "pipeline_bundle.h"
#include "pipeline_manifest.h"
#include "pipeline_metadata.h"
#include "target_binding_plan.h"

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vernon::runtime::ad {
class Executable;
class AutodiffMemoryPolicy;
size_t autodiffMemoryContextLimit(const std::shared_ptr<AutodiffMemoryPolicy> &policy);
} // namespace vernon::runtime::ad
namespace vernon::runtime::program {
struct ResolvedProgram;
} // namespace vernon::runtime::program
struct VernonProgramTopology;
// Internal definitions for the opaque C ABI handles.
struct VernonRuntimeContext {
    VernonRuntimeBackend backend{VERNON_RUNTIME_CPU};
    size_t liveBundles{};
    size_t livePipelines{};
    size_t liveContextLeases{};
    void *backendState{};
    void (*destroyBackendState)(void *){};
    bool borrowedRhiDevice{};
    VernonRhiDevice rhiDevice{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    std::shared_ptr<vernon::runtime::ad::AutodiffMemoryPolicy> autodiffMemoryPolicy;
    std::mutex cpuEntriesMutex;
    std::unordered_map<std::string, std::pair<VernonCpuEntryPoint, size_t>> cpuEntries;
    std::mutex referencedRhiBuffersMutex;
    std::map<std::pair<uint64_t, uint64_t>, VernonRhiBuffer> referencedRhiBuffers;
};

namespace vernon::runtime {

class ContextLease {
public:
    explicit ContextLease(VernonRuntimeContext &context) : context_(&context) { ++context_->liveContextLeases; }
    ContextLease(const ContextLease &) = delete;
    ContextLease &operator=(const ContextLease &) = delete;
    ~ContextLease() { --context_->liveContextLeases; }

    VernonRuntimeContext &get() const { return *context_; }

private:
    VernonRuntimeContext *context_;
};

inline std::shared_ptr<ContextLease> acquireContextLease(VernonRuntimeContext &context) {
    return std::make_shared<ContextLease>(context);
}

std::string &invocationDiagnostic(VernonRuntimeContext &context);
const std::string *currentInvocationDiagnostic(const VernonRuntimeContext &context);
void clearInvocationDiagnostic(const VernonRuntimeContext &context);

class RuntimeDiagnosticScope {
public:
    explicit RuntimeDiagnosticScope(const VernonRuntimeContext *context) noexcept;
    RuntimeDiagnosticScope(const RuntimeDiagnosticScope &) = delete;
    RuntimeDiagnosticScope &operator=(const RuntimeDiagnosticScope &) = delete;
    ~RuntimeDiagnosticScope() noexcept;

private:
    const VernonRuntimeContext *context_;
};

} // namespace vernon::runtime

struct VernonSubmission {
    std::shared_ptr<vernon::runtime::ContextLease> contextLease;
    VernonSubmissionState state{VERNON_SUBMISSION_PENDING};
    VernonStatus status{VERNON_STATUS_OK};
    VernonRhiDevice device{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    VernonRhiCompletion completion{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
};

template <typename State, typename Handle> State &runtimeBackendState(Handle &handle) {
    return *static_cast<State *>(handle.backendState);
}

template <typename State, typename Handle> const State &runtimeBackendState(const Handle &handle) {
    return *static_cast<const State *>(handle.backendState);
}

template <typename State, typename Handle> void installRuntimeBackendState(Handle &handle, State *state) {
    handle.backendState = state;
    handle.destroyBackendState = [](void *value) { delete static_cast<State *>(value); };
}

template <typename Handle> void destroyRuntimeBackendState(Handle &handle) {
    if (handle.destroyBackendState)
        handle.destroyBackendState(handle.backendState);
    handle.backendState = nullptr;
    handle.destroyBackendState = nullptr;
}

struct VernonProgramBundle {
    VernonRuntimeContext *context{};
    std::string id;
    std::unordered_map<std::string, vernon::runtime::Stage> stages;
    std::vector<vernon::runtime::Variant> variants;
    std::optional<vernon::runtime::AutodiffManifest> autodiff;
};

struct VernonDifferentiatedProgram {
    std::shared_ptr<vernon::runtime::ad::Executable> executable;
    std::vector<vernon::runtime::AutodiffDerivativeGroup> derivativeGroups;
};

struct VernonProgramExecutable {
    VernonProgramExecutable() = default;
    VernonProgramExecutable(const VernonProgramExecutable &) = delete;
    VernonProgramExecutable &operator=(const VernonProgramExecutable &) = delete;
    VernonProgramExecutable(VernonProgramExecutable &&) noexcept;
    VernonProgramExecutable &operator=(VernonProgramExecutable &&) noexcept;
    ~VernonProgramExecutable();

    VernonRuntimeContext *context{};
    vernon::runtime::Variant variant;
    VernonLaunchSize workgroupSize{1, 1, 1};
    vernon::runtime::DispatchContract dispatchContract;
    std::vector<vernon::runtime::TensorViewWriteFootprint> readFootprints;
    std::vector<vernon::runtime::TensorViewWriteFootprint> writeFootprints;
    std::shared_ptr<VernonProgramTopology> topology;
    std::optional<VernonDifferentiatedProgram> differentiated;
    void *backendState{};
    void (*destroyBackendState)(void *){};
};

struct VernonProgramStageBinding {
    uint32_t value{};
    std::optional<size_t> leaf;
    std::optional<vernon::runtime::program::TargetBinding> target;
};

struct VernonResolvedProgramStage {
    std::unique_ptr<VernonProgramExecutable> pipeline;
    std::vector<VernonProgramStageBinding> bindings;
    vernon::runtime::program::DispatchMapping dispatchMapping{vernon::runtime::program::DispatchMapping::StaticGrid};
};

struct VernonProgramTopology {
    ~VernonProgramTopology();

    std::shared_ptr<const vernon::runtime::program::ResolvedProgram> resolvedProgram;
    // Stable leaf/path backing for C-ABI reflection. Slot identity, role,
    // category, access, shape and ownership remain exclusively in ProgramABI.
    std::vector<vernon::runtime::ValueLayout> boundaryLayoutViews;
    std::vector<uint32_t> residualValues;
    std::vector<VernonResolvedProgramStage> stages;
    std::unordered_map<std::string, size_t> stageIndices;
    std::optional<uint64_t> programCheckpointMemoryBudget;
    std::string programCheckpointPolicy;
};

inline VernonProgramExecutable::VernonProgramExecutable(VernonProgramExecutable &&) noexcept = default;
inline VernonProgramExecutable &VernonProgramExecutable::operator=(VernonProgramExecutable &&) noexcept = default;
inline VernonProgramExecutable::~VernonProgramExecutable() = default;

#endif
