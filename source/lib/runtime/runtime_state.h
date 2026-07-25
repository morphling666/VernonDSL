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

// Internal definitions for the opaque C ABI handles. Backend encoders share
// these objects but ownership remains with VernonRuntime's public lifecycle.
struct VernonRuntimeContext {
    VernonRuntimeBackend backend{VERNON_RUNTIME_CPU};
    std::string error;
    size_t liveBuffers{};
    size_t liveTextures{};
    size_t liveSamplers{};
    size_t liveBundles{};
    size_t livePipelines{};
    void *backendState{};
    void (*destroyBackendState)(void *){};
    bool borrowedRhiDevice{};
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

struct VernonDeviceBuffer {
    VernonRuntimeContext *context{};
    size_t size{};
    size_t alignment{};
    void *backendState{};
    void (*destroyBackendState)(void *){};
};

struct VernonDeviceTexture {
    VernonRuntimeContext *context{};
    uint32_t width{};
    uint32_t height{};
    uint32_t depth{1};
    uint32_t mipLevels{1};
    VernonTextureDimension dimension{VERNON_TEXTURE_2D};
    VernonTextureFormat format{VERNON_TEXTURE_RGBA8_UNORM};
    void *backendState{};
    void (*destroyBackendState)(void *){};
};

struct VernonDeviceSampler {
    VernonRuntimeContext *context{};
    VernonSamplerDescriptor descriptor{};
    void *backendState{};
    void (*destroyBackendState)(void *){};
};

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
