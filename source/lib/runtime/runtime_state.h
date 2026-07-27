#ifndef VERNON_RUNTIME_RUNTIME_STATE_H
#define VERNON_RUNTIME_RUNTIME_STATE_H

#include "VernonRuntime.h"
#include "pipeline_bundle.h"
#include "pipeline_manifest.h"
#include "pipeline_metadata.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// Internal definitions for the opaque C ABI handles.
struct VernonRuntimeContext {
    VernonRuntimeBackend backend{VERNON_RUNTIME_CPU};
    std::string error;
    size_t liveBundles{};
    size_t livePipelines{};
    void *backendState{};
    void (*destroyBackendState)(void *){};
    bool borrowedRhiDevice{};
    VernonRhiDevice rhiDevice{static_cast<uint32_t>(VERNON_RHI_INVALID_HANDLE_INDEX), 0};
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

struct VernonPipelineBundle {
    VernonRuntimeContext *context{};
    std::string id;
    std::vector<std::string> features;
    std::unordered_map<std::string, vernon::runtime::Stage> stages;
    std::vector<vernon::runtime::Variant> variants;
};

struct VernonLoadedPipeline {
    VernonRuntimeContext *context{};
    vernon::runtime::Variant variant;
    void *backendState{};
    void (*destroyBackendState)(void *){};
};

#endif
