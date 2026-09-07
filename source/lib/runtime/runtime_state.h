#ifndef VERNON_RUNTIME_RUNTIME_STATE_H
#define VERNON_RUNTIME_RUNTIME_STATE_H

#include "VernonRuntime.h"
#include "program_execution_manifest.h"

#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vernon::runtime::ad {
class CanonicalProgramExecution;
class AutodiffMemoryPolicy;
size_t autodiffMemoryContextLimit(const std::shared_ptr<AutodiffMemoryPolicy> &policy);
} // namespace vernon::runtime::ad
namespace vernon::runtime::program {
struct ResolvedProgram;
struct ResolvedExecutionPlan;
} // namespace vernon::runtime::program
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

namespace vernon::runtime {
// One fully validated deployment variant. Loading parses the immutable
// descriptor; resolving copies the Program into a fresh physical executable.
struct ProgramVariantDeployment {
    std::vector<std::string> key;
    program::Program program;
    program::ArtifactSystem artifactSystem;
};

} // namespace vernon::runtime

struct VernonProgramBundle {
    VernonRuntimeContext *context{};
    std::string id;
    std::vector<vernon::runtime::ProgramVariantDeployment> deployments;
    std::filesystem::path bundleRoot;
};

struct CanonicalProgramAutodiffState {
    std::shared_ptr<vernon::runtime::ad::CanonicalProgramExecution> canonicalExecution;
    std::vector<vernon::runtime::AutodiffDerivativeGroup> derivativeGroups;
    std::optional<uint64_t> checkpointMemoryBudget;
    std::string checkpointPolicy;
};

struct VernonProgramExecutable {
    VernonProgramExecutable(VernonRuntimeContext &runtime,
                            std::shared_ptr<const vernon::runtime::program::ResolvedExecutionPlan> plan);
    VernonProgramExecutable(const VernonProgramExecutable &) = delete;
    VernonProgramExecutable &operator=(const VernonProgramExecutable &) = delete;
    VernonProgramExecutable(VernonProgramExecutable &&) = delete;
    VernonProgramExecutable &operator=(VernonProgramExecutable &&) = delete;
    ~VernonProgramExecutable() = default;

    VernonRuntimeContext *context;
    const std::shared_ptr<const vernon::runtime::program::ResolvedExecutionPlan> executionPlan;
    CanonicalProgramAutodiffState programAutodiff;
};

inline VernonProgramExecutable::VernonProgramExecutable(
    VernonRuntimeContext &runtime, std::shared_ptr<const vernon::runtime::program::ResolvedExecutionPlan> plan)
    : context(&runtime), executionPlan(std::move(plan)) {
    if (!executionPlan)
        throw std::invalid_argument("ProgramExecutable requires a resolved execution plan");
}

#endif
