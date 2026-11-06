#ifndef VERNON_RUNTIME_RUNTIME_STATE_H
#define VERNON_RUNTIME_RUNTIME_STATE_H

#include "VernonLifecycle.hpp"
#include "VernonRuntime.h"
#include "autodiff/autodiff_metadata.h"
#include "program_execution_manifest.h"
#include "runtime_lifecycle.h"

#include <nlohmann/json_fwd.hpp>

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace vernon::runtime::program {
struct ResolvedProgram;
struct ResolvedExecutionPlan;
} // namespace vernon::runtime::program
namespace vernon::hal {
class Device;
} // namespace vernon::hal
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
    uint64_t diagnosticGeneration{};
    VernonRuntimeBackend backend{VERNON_RUNTIME_CPU};
    std::atomic<uint64_t> nextProgramGraphId{1};
    void *backendState{};
    void (*destroyBackendState)(void *){};
    std::mutex cpuEntriesMutex;
    std::unordered_map<std::string, std::pair<VernonCpuEntryPoint, size_t>> cpuEntries;
    // The HAL device linked schedules record on. A CUDA context is created over it; on other backends the first
    // schedule creates it, and on the CPU it shares the backend's scheduler.
    std::mutex halDeviceMutex;
    std::shared_ptr<vernon::hal::Device> halDevice;
};

namespace vernon::runtime {

std::string &invocationDiagnostic(VernonRuntimeContext &context);
const std::string *currentInvocationDiagnostic(const VernonRuntimeContext &context);
VernonStringView currentInvocationDiagnosticView(const VernonRuntimeContext &context) noexcept;
void clearInvocationDiagnostic(const VernonRuntimeContext &context) noexcept;

class RuntimeDiagnosticScope {
public:
    explicit RuntimeDiagnosticScope(const VernonRuntimeContext *context) noexcept;
    RuntimeDiagnosticScope(const RuntimeDiagnosticScope &) = delete;
    RuntimeDiagnosticScope &operator=(const RuntimeDiagnosticScope &) = delete;
    ~RuntimeDiagnosticScope() noexcept;

private:
    const VernonRuntimeContext *context_;
    uint64_t generation_{};
    void *state_{};
};

} // namespace vernon::runtime

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
    Type,
    Format,
};

struct ProgramSpecialization {
    std::string name;
    ProgramSpecializationKind kind{ProgramSpecializationKind::Bool};
    // A Type value holds the canonical type identity string.
    std::variant<bool, int32_t, uint32_t, float, double, std::string, VernonTextureFormat> value{false};

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
    // The variant's `program` JSON, from which a linked schedule builds its fragments.
    std::shared_ptr<const nlohmann::json> source;
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
    std::vector<vernon::runtime::AutodiffDerivativeGroup> derivativeGroups;
};

struct VernonProgramExecutable;
struct RuntimeGraphicsPipelineState;

// A composed Program graph: one schedule node per Program node, in run order.
struct ProgramGraphComposition {
    // A boundary slot of a node's own Program, by the node's index in `nodes`.
    struct Site {
        uint32_t node{};
        uint32_t slot{};
    };
    struct Connection {
        // A Value edge's source then its destinations, or a Storage chain's versions in order.
        std::vector<Site> sites;
        bool storage{};
    };
    struct Export {
        Site site;
        // The composed executable's boundary slot.
        uint32_t boundary{};
    };
    // A graphics Node of a node's forward graph. Its composed id counts the forward Nodes of every earlier node, as
    // the linked schedule numbers the Nodes of its fragments.
    struct GraphicsNode {
        uint32_t id{};
        uint32_t node{};
        uint32_t local{};
    };

    std::vector<std::shared_ptr<VernonProgramExecutable>> nodes;
    // Per node, the label its schedule Nodes are named under; empty keeps the node's own names.
    std::vector<std::string> labels;
    std::vector<Connection> connections;
    std::vector<Export> exports;
    std::vector<GraphicsNode> graphicsNodes;
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
    std::vector<uint32_t> publicParameterSlots;
    std::unordered_set<uint32_t> bindableParameterSlots;
    std::unordered_map<uint64_t, std::vector<uint32_t>> bindingAliases;
    CanonicalProgramAutodiffState programAutodiff;
    // The `program` JSON of a bundle variant's executable; null for a composed graph, which links per node.
    std::shared_ptr<const nlohmann::json> programSource;
    // Set for a composed graph resolved on the CPU or CUDA.
    std::shared_ptr<const ProgramGraphComposition> composition;
    // Pipeline 0 of each recorded graphics Node. The states hold no lease on this executable, so every handle,
    // binding set, or recording that keeps one alive must itself keep this executable alive.
    std::mutex passPipelineMutex;
    std::map<VernonProgramNodeId, std::shared_ptr<const RuntimeGraphicsPipelineState>> passPipelines;
};

inline VernonProgramExecutable::VernonProgramExecutable(
    VernonRuntimeContext &runtime, vernon::runtime::RuntimeChildLifecycle childLifecycle,
    vernon::OwnerRef executableInstanceOwner,
    std::shared_ptr<const vernon::runtime::program::ResolvedExecutionPlan> plan) noexcept
    : lifecycle(std::move(childLifecycle)), instanceOwner(std::move(executableInstanceOwner)), context(&runtime),
      executionPlan(std::move(plan)) {}

#endif
