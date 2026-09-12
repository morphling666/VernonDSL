#ifndef VERNON_RUNTIME_RUNTIME_STATE_H
#define VERNON_RUNTIME_RUNTIME_STATE_H

#include "VernonLifecycle.hpp"
#include "VernonRuntime.h"
#include "program_execution_manifest.h"
#include "runtime_lifecycle.h"

#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
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
    [[nodiscard]] static vernon::Result<std::unique_ptr<VernonRuntimeContext>, vernon::RuntimeError> create() noexcept {
        using Result = vernon::Result<std::unique_ptr<VernonRuntimeContext>, vernon::RuntimeError>;
        auto ownerControl = vernon::OwnerControlBlock::create();
        if (ownerControl.isErr())
            return Result{vernon::err(vernon::toRuntimeError(std::move(ownerControl).error()))};
        auto operationControl = vernon::OperationControlBlock::create();
        if (operationControl.isErr())
            return Result{vernon::err(vernon::toRuntimeError(std::move(operationControl).error()))};
        auto context = vernon::tryMakeUnique<VernonRuntimeContext>(std::move(ownerControl).value(),
                                                                   std::move(operationControl).value());
        if (context.isErr())
            return Result{vernon::err(vernon::toRuntimeError(context.error(), {"create_runtime_context", 0, 0}))};
        return Result{vernon::ok(std::move(context).value())};
    }

    VernonRuntimeContext(vernon::OwnerRef ownerControl, vernon::OperationRef operationControl) noexcept
        : owner(std::move(ownerControl)), operations(std::move(operationControl)) {}
    VernonRuntimeContext(const VernonRuntimeContext &) = delete;
    VernonRuntimeContext &operator=(const VernonRuntimeContext &) = delete;

    vernon::OwnerRef owner;
    vernon::OperationRef operations;
    VernonRuntimeBackend backend{VERNON_RUNTIME_CPU};
    std::atomic<uint64_t> nextProgramGraphId{1};
    void *backendState{};
    void (*destroyBackendState)(void *){};
    VernonRhiDevice rhiDevice{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
    vernon::Option<vernon::ChildLease> rhiDeviceLease;
    std::shared_ptr<vernon::runtime::ad::AutodiffMemoryPolicy> autodiffMemoryPolicy;
    std::mutex cpuEntriesMutex;
    std::unordered_map<std::string, std::pair<VernonCpuEntryPoint, size_t>> cpuEntries;
    std::mutex referencedRhiBuffersMutex;
    std::map<std::pair<uint64_t, uint64_t>, VernonRhiBuffer> referencedRhiBuffers;
};

namespace vernon::runtime {

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
    vernon::Option<vernon::runtime::RuntimeChildLifecycle> lifecycle;
    VernonRuntimeContext *context{};
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
enum class ProgramSpecializationKind {
    Bool,
    I32,
    U32,
    F32,
    F64,
};

struct ProgramSpecialization {
    std::string name;
    ProgramSpecializationKind kind{ProgramSpecializationKind::Bool};
    std::variant<bool, int32_t, uint32_t, float, double> value{false};

    bool operator==(const ProgramSpecialization &other) const {
        return name == other.name && kind == other.kind && value == other.value;
    }
    bool operator<(const ProgramSpecialization &other) const {
        return name != other.name ? name < other.name : std::tie(kind, value) < std::tie(other.kind, other.value);
    }
};

// One fully validated deployment variant. Loading parses the immutable
// descriptor; resolving copies the Program into a fresh physical executable.
struct ProgramVariantDeployment {
    std::vector<ProgramSpecialization> key;
    program::Program program;
    program::ArtifactSystem artifactSystem;
};

} // namespace vernon::runtime

struct VernonProgramBundle {
    explicit VernonProgramBundle(vernon::runtime::RuntimeChildLifecycle childLifecycle) noexcept
        : lifecycle(std::move(childLifecycle)) {}

    vernon::runtime::RuntimeChildLifecycle lifecycle;
    VernonRuntimeContext *context{};
    std::string id;
    std::string contentHash;
    std::vector<vernon::runtime::ProgramVariantDeployment> deployments;
    std::filesystem::path bundleRoot;
};

struct CanonicalProgramAutodiffState {
    std::shared_ptr<vernon::runtime::ad::CanonicalProgramExecution> canonicalExecution;
    std::vector<vernon::runtime::AutodiffDerivativeGroup> derivativeGroups;
    std::optional<uint64_t> checkpointMemoryBudget;
    std::string checkpointPolicy;
};

struct VernonProgramExecutable;

struct ProgramGraphNodeAutodiffState {
    std::shared_ptr<VernonProgramExecutable> executable;
    std::vector<uint32_t> globalValues;
    std::vector<uint32_t> globalStorages;
};

struct VernonProgramExecutable {
    VernonProgramExecutable(VernonRuntimeContext &runtime, vernon::runtime::RuntimeChildLifecycle childLifecycle,
                            vernon::OwnerRef instanceOwner,
                            std::shared_ptr<const vernon::runtime::program::ResolvedExecutionPlan> plan) noexcept;
    VernonProgramExecutable(const VernonProgramExecutable &) = delete;
    VernonProgramExecutable &operator=(const VernonProgramExecutable &) = delete;
    VernonProgramExecutable(VernonProgramExecutable &&) = delete;
    VernonProgramExecutable &operator=(VernonProgramExecutable &&) = delete;
    ~VernonProgramExecutable() = default;

    vernon::runtime::RuntimeChildLifecycle lifecycle;
    vernon::OwnerRef instanceOwner;
    VernonRuntimeContext *context;
    std::string id;
    uint64_t programGraphId{};
    const std::shared_ptr<const vernon::runtime::program::ResolvedExecutionPlan> executionPlan;
    std::map<uint64_t, std::vector<uint32_t>> programGraphBoundarySlots;
    std::map<uint32_t, std::vector<uint32_t>> programGraphStorageSlots;
    std::map<uint64_t, VernonProgramGraphicsControlsView> programGraphGraphicsControls;
    std::map<VernonProgramNodeId, ProgramGraphNodeAutodiffState> programGraphNodeAutodiff;
    CanonicalProgramAutodiffState programAutodiff;
};

inline VernonProgramExecutable::VernonProgramExecutable(
    VernonRuntimeContext &runtime, vernon::runtime::RuntimeChildLifecycle childLifecycle,
    vernon::OwnerRef executableInstanceOwner,
    std::shared_ptr<const vernon::runtime::program::ResolvedExecutionPlan> plan) noexcept
    : lifecycle(std::move(childLifecycle)), instanceOwner(std::move(executableInstanceOwner)), context(&runtime),
      executionPlan(std::move(plan)) {}

#endif
