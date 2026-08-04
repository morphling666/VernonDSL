#ifndef VERNON_RUNTIME_RUNTIME_STATE_H
#define VERNON_RUNTIME_RUNTIME_STATE_H

#include "VernonRuntime.h"
#include "pipeline_bundle.h"
#include "pipeline_manifest.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vernon::runtime::ad {
class Executable;
}
namespace vernon::runtime {
class CompiledAutodiffGraph;
}

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
    explicit RuntimeDiagnosticScope(const VernonRuntimeContext *context);
    RuntimeDiagnosticScope(const RuntimeDiagnosticScope &) = delete;
    RuntimeDiagnosticScope &operator=(const RuntimeDiagnosticScope &) = delete;
    ~RuntimeDiagnosticScope();

private:
    const VernonRuntimeContext *context_;
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

struct VernonPipelineBundle {
    VernonRuntimeContext *context{};
    std::string id;
    std::vector<std::string> features;
    std::unordered_map<std::string, vernon::runtime::Stage> stages;
    std::vector<vernon::runtime::Variant> variants;
    std::optional<vernon::runtime::AutodiffManifest> autodiff;
};

struct VernonLoadedAutodiff {
    std::shared_ptr<vernon::runtime::ad::Executable> executable;
    std::shared_ptr<vernon::runtime::CompiledAutodiffGraph> immediateGraph;
};

struct VernonLoadedPipeline {
    VernonRuntimeContext *context{};
    vernon::runtime::Variant variant;
    std::optional<VernonLoadedAutodiff> autodiff;
    void *backendState{};
    void (*destroyBackendState)(void *){};
};

#endif
